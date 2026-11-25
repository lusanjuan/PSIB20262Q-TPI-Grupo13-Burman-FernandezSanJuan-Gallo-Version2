# descriptores.py
# Funciones para el paso 6: huella para buscar duplicados, los 5 descriptores
# de la lesión y la elipse equivalente de una máscara.
# Se importa así:  from descriptores import huella, mancha_del_clic, descriptores, iou_elipse
#
# Los descriptores se calculan sobre la imagen limpia SIN filtrar: el paso 4
# mostró que Lee no mejora la textura, y así medimos los grises originales.

import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage.draw import ellipse

from filtros import abanico


# ---------------------------------------------------------------------------
# Huella para buscar duplicados.
# Reducimos la imagen a 64x64 y le restamos su versión suavizada (gaussiano
# σ = 4). Lo que queda son los detalles (bordes, estructuras), sin el brillo
# general: dos fotos de la misma lesión tienen huellas parecidas aunque una
# sea más clara que la otra.
# ---------------------------------------------------------------------------
def huella(img):
    chica = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(float)
    return chica - ndimage.gaussian_filter(chica, 4)


# ---------------------------------------------------------------------------
# Si la máscara tiene varias manchas, nos quedamos con la que contiene el clic.
# ---------------------------------------------------------------------------
def mancha_del_clic(mascara, clic):
    etiquetas, _ = ndimage.label(mascara)
    if etiquetas[clic] == 0:
        return np.zeros(mascara.shape, dtype=bool)
    return etiquetas == etiquetas[clic]


# ---------------------------------------------------------------------------
# GLCM (matriz de co-ocurrencia de grises), igual que en el paso 4:
# 32 niveles fijos (nivel = gris // 8), distancia 1, 4 ángulos (0°, 45°, 90°,
# 135°), simétrica. Los píxeles fuera de la región van a un nivel extra (32)
# que después se descarta: así solo cuentan los pares con los dos píxeles
# dentro de la región. Se normaliza para que cada matriz sume 1.
# Homogeneidad = suma de P(i, j) / (1 + (i − j)²): alta cuando los vecinos
# tienen grises parecidos (textura pareja).
# ---------------------------------------------------------------------------
def homogeneidad_glcm(imagen, region):
    niveles = np.clip(np.floor(imagen / 8), 0, 31).astype(np.uint8)
    niveles[~region] = 32
    angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    P = graycomatrix(niveles, distances=[1], angles=angulos, levels=33, symmetric=True)
    P = P[:32, :32, :, :].astype(float)
    P = P / P.sum(axis=(0, 1), keepdims=True)
    return graycoprops(P, "homogeneity").mean()


# ---------------------------------------------------------------------------
# Los 5 descriptores de una lesión (img = imagen limpia sin filtrar,
# lesion = máscara de una sola mancha). Devuelve un diccionario.
#
# 1) Circularidad = 4π·área / perímetro². Vale 1 en un círculo y baja cuanto
#    más irregular es el contorno. Las malignas suelen tener bordes irregulares
#    (espiculados), así que deberían dar menos.
# 2) Alto/ancho = alto ÷ ancho del rectángulo que contiene la lesión. Mayor que
#    1 = la lesión es "más alta que ancha" (crece atravesando los planos del
#    tejido), un signo clásico de malignidad en ecografía.
# 3) Ecogenicidad = media de gris de la lesión ÷ media del anillo de 5 a 20 px
#    por fuera (dentro del abanico, como en el paso 4). Menor que 1 = lesión
#    más oscura que el tejido (hipoecoica).
# 4) Sombra = media de un rectángulo debajo de la lesión (mismo ancho que la
#    lesión, alto = mitad del alto de la lesión) ÷ media de dos rectángulos
#    iguales a los costados, a la misma profundidad. Menor que 1 = el sonido se
#    atenúa al atravesar la lesión (sombra acústica, más común en malignas).
#    Solo cuentan los píxeles dentro del abanico; si alguna de las dos zonas
#    queda vacía (lesión pegada al fondo o al costado), vale NaN.
# 5) Homogeneidad GLCM relativa = homogeneidad de la lesión ÷ la del anillo.
#    Mayor que 1 = la lesión es más pareja por dentro que el tejido de alrededor.
# ---------------------------------------------------------------------------
def descriptores(img, lesion):
    props = regionprops(lesion.astype(np.uint8))[0]
    fila0, columna0, fila1, columna1 = props.bbox       # fila1 y columna1 no incluidas
    alto, ancho = fila1 - fila0, columna1 - columna0

    distancia = ndimage.distance_transform_edt(~lesion)
    dentro = abanico(img)
    anillo = (distancia >= 5) & (distancia <= 20) & dentro

    # Rectángulos para la sombra
    debajo = np.zeros(img.shape, dtype=bool)
    costados = np.zeros(img.shape, dtype=bool)
    filas_sombra = slice(fila1, fila1 + alto // 2)
    debajo[filas_sombra, columna0:columna1] = True
    costados[filas_sombra, max(columna0 - ancho, 0):columna0] = True
    costados[filas_sombra, columna1:columna1 + ancho] = True
    debajo &= dentro
    costados &= dentro
    if debajo.any() and costados.any():
        sombra = img[debajo].mean() / img[costados].mean()
    else:
        sombra = np.nan

    # Recorte alrededor de lesión + anillo, para que la GLCM sea más rápida
    filas, columnas = np.where(lesion | anillo)
    recorte = (slice(filas.min(), filas.max() + 1), slice(columnas.min(), columnas.max() + 1))
    homogeneidad = (homogeneidad_glcm(img[recorte], lesion[recorte])
                    / homogeneidad_glcm(img[recorte], anillo[recorte]))

    return {
        "circularidad": 4 * np.pi * props.area / props.perimeter ** 2,
        "alto/ancho": alto / ancho,
        "ecogenicidad": img[lesion].mean() / img[anillo].mean(),
        "sombra": sombra,
        "homogeneidad": homogeneidad,
    }


# ---------------------------------------------------------------------------
# IoU de una máscara con su elipse equivalente (la elipse con los mismos
# momentos: mismo centro, mismos ejes y misma orientación; regionprops).
# IoU = |A∩B| / |A∪B|. Cerca de 1 = la máscara es casi una elipse (contorno
# liso, dibujado "a mano alzada" redondeado); más bajo = contorno irregular.
# ---------------------------------------------------------------------------
def iou_elipse(lesion):
    props = regionprops(lesion.astype(np.uint8))[0]
    fila_c, columna_c = props.centroid
    elipse = np.zeros(lesion.shape, dtype=bool)
    filas, columnas = ellipse(fila_c, columna_c, props.axis_major_length / 2, props.axis_minor_length / 2,
                              shape=lesion.shape, rotation=props.orientation)
    elipse[filas, columnas] = True
    return (lesion & elipse).sum() / (lesion | elipse).sum()
