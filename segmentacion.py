# segmentacion.py
# Métodos para segmentar la lesión a partir de un clic dentro de ella.
# Se importa así:  from segmentacion import segmentar, ventana_alrededor, mapas_ventana, METODOS
#
# Regla general: la máscara manual NO se usa acá. Estos métodos solo reciben
# la imagen y el clic. La máscara se usa en paso6_segmentacion.py únicamente
# para simular el clic y para evaluar.

import numpy as np
import cv2
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from skimage.filters import threshold_otsu
from skimage.morphology import disk

from filtros import filtro_mediana
from mapas import mapa_media, mapa_coef_variacion, mapas_glcm

METODOS = ["RG mediana", "RG media local", "Otsu media local", "k-means media", "k-means 3 mapas"]
USA_TOLERANCIA = ["RG mediana", "RG media local"]


def ventana_alrededor(forma, clic, lado):
    """Ventana cuadrada de `lado` px centrada en el clic (recortada si se sale de la imagen)."""
    y, x = clic
    medio = lado // 2
    return (slice(max(y - medio, 0), min(y + medio, forma[0])),
            slice(max(x - medio, 0), min(x + medio, forma[1])))


def mapas_ventana(cruda):
    """Mapas que usan los métodos, calculados solo dentro de la ventana (con mapas.py):
    - imagen con mediana 7x7;
    - media local 15 (sobre la mediana), coeficiente de variación 31 y contraste
      de la GLCM 31 (sobre la imagen sin filtrar): los 3 mapas elegidos en el paso 5."""
    filtrada = filtro_mediana(cruda)
    return {
        "mediana": filtrada,
        "media local": mapa_media(filtrada, 15),
        "coef. de variación": mapa_coef_variacion(cruda, 31),
        "contraste": mapas_glcm(cruda, 31)["contraste"],
    }


# ---------------------------------------------------------------------------
# Region growing (crecimiento de regiones).
# Arranca con el píxel del clic y agrega los vecinos (8-vecindad: los 8 píxeles
# de alrededor) cuyo gris esté a menos de `tolerancia` del promedio actual de
# la región. El promedio se actualiza y se repite hasta que no entra nadie más.
# Para que sea rápido, en cada vuelta se agregan de una vez todos los vecinos
# del borde que cumplen, y recién después se recalcula el promedio.
# La tolerancia está en niveles de gris (escala 0–255).
# ---------------------------------------------------------------------------
def region_growing(img, clic, tolerancia):
    region = np.zeros(img.shape, dtype=np.uint8)
    region[clic] = 1
    vecindad = np.ones((3, 3), dtype=np.uint8)   # 8-vecindad
    while True:
        promedio = img[region == 1].mean()
        borde = (cv2.dilate(region, vecindad) == 1) & (region == 0)
        nuevos = borde & (np.abs(img - promedio) < tolerancia)
        if not nuevos.any():
            break
        region[nuevos] = 1
    return region == 1


# ---------------------------------------------------------------------------
# Otsu: elige automáticamente el umbral de brillo que mejor separa los píxeles
# en dos grupos (el que hace que cada grupo sea lo más parejo posible por
# dentro). Nos quedamos con el lado del umbral donde cae el clic.
# Sirve para comprobar si un umbral de brillo solo alcanza para segmentar.
# ---------------------------------------------------------------------------
def otsu_lado_clic(mapa, clic):
    umbral = threshold_otsu(mapa)
    if mapa[clic] > umbral:
        return mapa > umbral
    return mapa <= umbral


# ---------------------------------------------------------------------------
# k-means con 2 grupos: reparte los píxeles en 2 grupos de modo que cada píxel
# quede con el grupo de valores (características) más parecidos.
# Con varios mapas, cada uno se estandariza (resta su media y divide por su
# desvío) para que ninguno pese más solo por su escala.
# El grupo de la lesión es el grupo al que pertenece el clic.
# ---------------------------------------------------------------------------
def kmeans_lado_clic(lista_mapas, clic):
    X = np.column_stack([m.ravel() for m in lista_mapas]).astype(float)
    desvio = X.std(axis=0)
    desvio[desvio == 0] = 1
    X = (X - X.mean(axis=0)) / desvio
    _, etiquetas = kmeans2(X, 2, seed=13, minit="++")
    etiquetas = etiquetas.reshape(lista_mapas[0].shape)
    return etiquetas == etiquetas[clic]


# ---------------------------------------------------------------------------
# Refinamiento, igual para todos los métodos:
#   - rellenar agujeros (la lesión es una mancha llena);
#   - apertura con un disco de radio 3 (borra puentes finos y salientes chicas);
#   - quedarse con la mancha que contiene el clic. Si la apertura borró el
#     clic, la segmentación queda vacía.
# ---------------------------------------------------------------------------
def refinar(mascara, clic):
    mascara = ndimage.binary_fill_holes(mascara)
    mascara = ndimage.binary_opening(mascara, structure=disk(3))
    etiquetas, _ = ndimage.label(mascara)
    if etiquetas[clic] == 0:
        return np.zeros(mascara.shape, dtype=bool)
    return etiquetas == etiquetas[clic]


def segmentar(metodo, mapas, clic, tolerancia=None):
    """Segmenta con uno de los 5 métodos, dentro de la ventana (mapas de mapas_ventana).
    `clic` está en coordenadas de la ventana. Devuelve la máscara ya refinada."""
    if metodo == "RG mediana":
        cruda = region_growing(mapas["mediana"], clic, tolerancia)
    elif metodo == "RG media local":
        cruda = region_growing(mapas["media local"], clic, tolerancia)
    elif metodo == "Otsu media local":
        cruda = otsu_lado_clic(mapas["media local"], clic)
    elif metodo == "k-means media":
        cruda = kmeans_lado_clic([mapas["media local"]], clic)
    elif metodo == "k-means 3 mapas":
        cruda = kmeans_lado_clic([mapas["media local"], mapas["coef. de variación"], mapas["contraste"]], clic)
    return refinar(cruda, clic)
