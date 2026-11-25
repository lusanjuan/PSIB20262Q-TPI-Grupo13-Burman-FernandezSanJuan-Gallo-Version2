# filtros.py
# Filtro de Lee para suavizar el speckle (el filtro adaptativo para speckle
# visto en clase), y las funciones que necesita.
# Se importa así:  from filtros import filtro_lee, varianza_ruido, abanico
#
# Ventana de 7x7: la misma con la que el paso 3 (bloque 5) midió la varianza
# local de cada imagen. Lee compara la varianza de cada ventana con la del
# ruido, así que las dos tienen que medirse con la misma ventana.

import numpy as np
from scipy import ndimage

VENTANA = 7


# ---------------------------------------------------------------------------
# Abanico de la sonda y varianza del ruido de cada imagen.
# Abanico = lo que no es fondo negro (gris > 0), con los huecos rellenados
# (dentro del tejido oscuro hay píxeles sueltos en 0 que no son fondo).
# Varianza del ruido = mediana de la varianza local 7x7, solo en ventanas
# enteras dentro del abanico y sin marcas (igual que el bloque 5 del paso 3).
# Es un valor por imagen: el paso 3 mostró que cambia mucho de una imagen a otra.
# ---------------------------------------------------------------------------
def abanico(img):
    return ndimage.binary_fill_holes(ndimage.binary_closing(img > 0, np.ones((5, 5))))


def media_y_varianza_local(img):
    x = img.astype(float)
    media = ndimage.uniform_filter(x, VENTANA)
    varianza = ndimage.uniform_filter(x ** 2, VENTANA) - media ** 2
    return media, np.maximum(varianza, 0)


def varianza_ruido(img, mascara_marcas):
    _, varianza = media_y_varianza_local(img)
    valido = abanico(img) & ~mascara_marcas
    valido = ndimage.binary_erosion(valido, np.ones((VENTANA, VENTANA)))
    return np.median(varianza[valido])


# ---------------------------------------------------------------------------
# Filtro de Lee (ventana 7x7).
# Con la media local m y la varianza local v de cada ventana:
#     k = max(0, (v − var_ruido) / v)        salida = m + k · (píxel − m)
# - Si la ventana varía lo mismo que el ruido (v ≈ var_ruido), k ≈ 0 y la
#   salida es la media: suaviza mucho, porque esa variación es solo speckle.
# - Si la ventana varía mucho más que el ruido (un borde), k ≈ 1 y la salida
#   es el píxel original: casi no toca los bordes.
# Devuelve una imagen float (sin redondear), del mismo tamaño que la entrada.
# ---------------------------------------------------------------------------
def filtro_lee(img, var_ruido):
    m, v = media_y_varianza_local(img)
    k = np.zeros_like(v)
    hay_variacion = v > 0
    k[hay_variacion] = np.maximum(0, (v[hay_variacion] - var_ruido) / v[hay_variacion])
    return m + k * (img.astype(float) - m)
