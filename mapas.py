# mapas.py
# Funciones para calcular "mapas" de brillo y de textura.
# Se importa así:  from mapas import calcular_mapas, recorte_lesion, NOMBRES_MAPAS
#
# Qué es un mapa: una imagen del mismo tamaño que la original donde cada píxel
# vale una medida calculada en una ventana alrededor de ese píxel.
# Ejemplo chico, mapa de media local con ventana de 3x3:
#
#     imagen:              mapa de media 3x3 (píxel del centro):
#     10 10 10 10 10
#     10 10 10 10 10       la ventana alrededor del píxel central (fila 2, col 2)
#     10 10 90 10 10       tiene ocho 10 y un 90 -> media = (8·10 + 90) / 9 ≈ 18,9
#     10 10 10 10 10       un píxel de la esquina de la ventana (fila 1, col 1)
#     10 10 10 10 10       también ve el 90 -> 18,9; uno lejos (fila 0, col 4) -> 10
#
# Lo mismo con cualquier medida: varianza, gradiente, propiedades de la GLCM...
# Una ventana más grande da un mapa más estable pero más borroso.

import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize

VENTANAS = [7, 15, 31]
MARGEN_RECORTE = 40
PASO_GRILLA = 4

PROPIEDADES_GLCM = ["contraste", "homogeneidad", "energía", "correlación", "entropía"]
NOMBRE_SKIMAGE = {"contraste": "contrast", "homogeneidad": "homogeneity", "energía": "energy",
                  "correlación": "correlation", "entropía": "entropy"}
NOMBRES_MAPAS = ["intensidad", "media local", "varianza local", "coef. de variación",
                 "gradiente"] + PROPIEDADES_GLCM


def recorte_lesion(lesion):
    """Caja de la máscara de la lesión + 40 px para cada lado (sin salirse de la imagen)."""
    filas, columnas = np.where(lesion)
    alto, ancho = lesion.shape
    return (slice(max(filas.min() - MARGEN_RECORTE, 0), min(filas.max() + MARGEN_RECORTE + 1, alto)),
            slice(max(columnas.min() - MARGEN_RECORTE, 0), min(columnas.max() + MARGEN_RECORTE + 1, ancho)))


# ---------------------------------------------------------------------------
# Mapas de brillo y de textura simples (con filtros de ventana).
# ---------------------------------------------------------------------------
def mapa_media(img, ventana):
    return ndimage.uniform_filter(img.astype(float), ventana)


def mapa_varianza(img, ventana):
    x = img.astype(float)
    media = ndimage.uniform_filter(x, ventana)
    return np.maximum(ndimage.uniform_filter(x ** 2, ventana) - media ** 2, 0)


def mapa_coef_variacion(img, ventana):
    """Desvío / media. Si la media es 0, vale 0."""
    media = mapa_media(img, ventana)
    desvio = np.sqrt(mapa_varianza(img, ventana))
    cv = np.zeros_like(media)
    np.divide(desvio, media, out=cv, where=media > 0)
    return cv


def mapa_gradiente(img, ventana):
    """Módulo del gradiente de Sobel, promediado en la ventana."""
    x = img.astype(float)
    gradiente = np.hypot(ndimage.sobel(x, axis=0), ndimage.sobel(x, axis=1))
    return ndimage.uniform_filter(gradiente, ventana)


# ---------------------------------------------------------------------------
# Mapas de GLCM.
# Parámetros de la GLCM:
#   - 16 niveles fijos: nivel = gris // 16 (0 a 15);
#   - distancia 1;
#   - ángulos 0°, 45°, 90° y 135°;
#   - simétrica (cada par se cuenta como (i, j) y como (j, i)) y normalizada;
#   - cada propiedad se calcula en cada ángulo y se promedian los 4 valores.
# Propiedades:
#   - contraste    = Σ P(i,j)·(i − j)²     (alto = vecinos muy distintos)
#   - homogeneidad = Σ P(i,j) / (1 + (i − j)²)  (alto = vecinos parecidos)
#   - energía      = raíz(Σ P(i,j)²)        (alto = pocos pares que se repiten mucho)
#   - correlación  = cuánto predice el gris de un píxel el de su vecino
#   - entropía     = −Σ P(i,j)·log P(i,j)   (alto = pares muy variados, desorden)
# Para que sea rápido, la GLCM se calcula solo en una grilla cada 4 px y
# después el mapa se agranda (interpolando) al tamaño del recorte.
# ---------------------------------------------------------------------------
def mapas_glcm(img, ventana):
    niveles = (img // 16).astype(np.uint8)
    medio = ventana // 2
    con_borde = np.pad(niveles, medio, mode="reflect")   # para que las ventanas del borde estén completas
    angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    filas_grilla = range(0, img.shape[0], PASO_GRILLA)
    columnas_grilla = range(0, img.shape[1], PASO_GRILLA)
    grilla = {p: np.zeros((len(filas_grilla), len(columnas_grilla))) for p in PROPIEDADES_GLCM}

    for i, y in enumerate(filas_grilla):
        for j, x in enumerate(columnas_grilla):
            # La ventana centrada en (y, x) de la imagen original empieza en (y, x) de la imagen con borde
            v = con_borde[y:y + ventana, x:x + ventana]
            P = graycomatrix(v, distances=[1], angles=angulos, levels=16, symmetric=True, normed=True)
            for p in PROPIEDADES_GLCM:
                grilla[p][i, j] = graycoprops(P, NOMBRE_SKIMAGE[p]).mean()

    # Agrandar la grilla al tamaño de la imagen (interpolación lineal)
    return {p: resize(grilla[p], img.shape, order=1, mode="edge") for p in PROPIEDADES_GLCM}


# ---------------------------------------------------------------------------
# Todos los mapas de una imagen.
# - Brillo, sobre la imagen filtrada con mediana 7x7: intensidad (sin ventana)
#   y media local.
# - Textura, sobre la imagen sin filtrar: varianza, coeficiente de variación,
#   gradiente y las 5 propiedades de la GLCM.
# Devuelve un diccionario {(nombre, ventana): mapa}; la intensidad lleva ventana 0.
# Si se pasa `ventanas_por_mapa` ({nombre: ventana}), calcula solo esa ventana de cada mapa.
# ---------------------------------------------------------------------------
def calcular_mapas(filtrada, cruda, ventanas_por_mapa=None):
    def ventanas(nombre):
        if ventanas_por_mapa is None:
            return VENTANAS
        return [ventanas_por_mapa[nombre]]

    mapas = {("intensidad", 0): filtrada.astype(float)}
    for w in ventanas("media local"):
        mapas[("media local", w)] = mapa_media(filtrada, w)
    for w in ventanas("varianza local"):
        mapas[("varianza local", w)] = mapa_varianza(cruda, w)
    for w in ventanas("coef. de variación"):
        mapas[("coef. de variación", w)] = mapa_coef_variacion(cruda, w)
    for w in ventanas("gradiente"):
        mapas[("gradiente", w)] = mapa_gradiente(cruda, w)

    ventanas_glcm = VENTANAS if ventanas_por_mapa is None else sorted({ventanas_por_mapa[p] for p in PROPIEDADES_GLCM})
    for w in ventanas_glcm:
        glcm = mapas_glcm(cruda, w)
        for p in PROPIEDADES_GLCM:
            if ventanas_por_mapa is None or ventanas_por_mapa[p] == w:
                mapas[(p, w)] = glcm[p]
    return mapas
