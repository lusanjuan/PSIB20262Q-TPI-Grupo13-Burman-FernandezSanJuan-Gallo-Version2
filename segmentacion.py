# segmentacion.py
# Métodos para segmentar la lesión a partir de un clic dentro de ella.
# Se importa así:  from segmentacion import segmentar, ventana_alrededor, METODOS
#
# Regla general: la máscara manual NO se usa acá. Estos métodos solo reciben
# la imagen y el clic. La máscara se usa en paso5_segmentacion.py únicamente
# para simular el clic y para evaluar.
#
# Los 3 métodos trabajan sobre la imagen filtrada con Lee (paso 4), recortada
# en una ventana cuadrada alrededor del clic:
#   a) Region growing -> familia: crecimiento de regiones
#   b) Otsu           -> familia: umbralización
#   c) k-means (2 grupos) -> familia: agrupamiento (clustering)

import numpy as np
import cv2
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from skimage.filters import threshold_otsu
from skimage.morphology import disk

METODOS = ["Region growing", "Otsu", "k-means"]
USA_TOLERANCIA = ["Region growing"]


def ventana_alrededor(forma, clic, lado):
    """Ventana cuadrada de `lado` px centrada en el clic (recortada si se sale de la imagen)."""
    y, x = clic
    medio = lado // 2
    return (slice(max(y - medio, 0), min(y + medio, forma[0])),
            slice(max(x - medio, 0), min(x + medio, forma[1])))


# ---------------------------------------------------------------------------
# a) Region growing (familia: crecimiento de regiones).
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
# b) Otsu (familia: umbralización).
# Elige automáticamente el umbral de brillo que mejor separa los píxeles de la
# ventana en dos grupos (el que hace que cada grupo sea lo más parejo posible
# por dentro). Nos quedamos con el lado del umbral donde cae el clic.
# Es un umbral global (uno solo para toda la ventana): no mira vecindad.
# ---------------------------------------------------------------------------
def otsu_lado_clic(img, clic):
    umbral = threshold_otsu(img)
    if img[clic] > umbral:
        return img > umbral
    return img <= umbral


# ---------------------------------------------------------------------------
# c) k-means con 2 grupos (familia: agrupamiento / clustering).
# Reparte los píxeles de la ventana en 2 grupos según su gris: cada píxel va
# al grupo cuyo promedio está más cerca, y los promedios se recalculan hasta
# que no cambian. El grupo de la lesión es el grupo al que pertenece el clic.
# Con una sola característica (el gris) se parece a un umbral, pero el corte
# sale de los promedios de los grupos y no de la varianza dentro de cada uno.
# ---------------------------------------------------------------------------
def kmeans_lado_clic(img, clic):
    X = img.reshape(-1, 1).astype(float)
    _, etiquetas = kmeans2(X, 2, seed=13, minit="++")
    etiquetas = etiquetas.reshape(img.shape)
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


def segmentar(metodo, img, clic, tolerancia=None):
    """Segmenta con uno de los 3 métodos. `img` es la ventana de la imagen
    filtrada con Lee y `clic` está en coordenadas de la ventana.
    Devuelve la máscara ya refinada."""
    if metodo == "Region growing":
        cruda = region_growing(img, clic, tolerancia)
    elif metodo == "Otsu":
        cruda = otsu_lado_clic(img, clic)
    elif metodo == "k-means":
        cruda = kmeans_lado_clic(img, clic)
    return refinar(cruda, clic)
