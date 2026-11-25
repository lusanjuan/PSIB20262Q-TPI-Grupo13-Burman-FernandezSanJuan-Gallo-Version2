# filtros.py
# Filtro de Lee para suavizar el speckle (el filtro adaptativo para speckle
# visto en clase), y las funciones que necesita.
# Se importa así:  from filtros import filtro_lee, ruido_para_lee, abanico
#
# Ventana de 7x7: la misma con la que el paso 3 midió la varianza del ruido.
# Lee compara la varianza de cada ventana con la del ruido, así que las dos
# tienen que medirse con la misma ventana.

import os
from functools import lru_cache

import numpy as np
from scipy import ndimage

VENTANA = 7

# Varianza del ruido que usa Lee (en los pasos 4 y 5, que la piden con
# ruido_para_lee). Cambiándola acá, cambia en los dos pasos a la vez.
#   True:  un solo valor, el que midió el paso 3 en parches de tejido homogéneo
#          con ventanas de 7x7. Es una medida del RUIDO: no depende de cuánta
#          estructura tenga cada imagen.
#   False: un valor por imagen, la mediana de su varianza local 7x7
#          (varianza_ruido, más abajo). Esa mediana mezcla ruido y estructura,
#          y como es la mediana, Lee promedia siempre la mitad de cada imagen.
USAR_RUIDO_PASO3 = True
RUTA_RUIDO_PASO3 = "outputs/paso3_speckle/varianza_ruido.txt"


# ---------------------------------------------------------------------------
# Abanico de la sonda y varianza del ruido.
# Abanico = lo que no es fondo negro (gris > 0), con los huecos rellenados
# (dentro del tejido oscuro hay píxeles sueltos en 0 que no son fondo).
# varianza_ruido(img, marcas) = mediana de la varianza local 7x7 de la imagen,
# solo en ventanas enteras dentro del abanico y sin marcas (un valor por imagen).
# ruido_para_lee(img, marcas) = el valor que usa Lee, según USAR_RUIDO_PASO3.
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


@lru_cache(maxsize=1)
def leer_ruido_paso3():
    if not os.path.exists(RUTA_RUIDO_PASO3):
        raise FileNotFoundError(f"No existe {RUTA_RUIDO_PASO3}: corré primero paso3_speckle.py")
    with open(RUTA_RUIDO_PASO3) as archivo:
        return float(archivo.read())


def ruido_para_lee(img, mascara_marcas):
    if USAR_RUIDO_PASO3:
        return leer_ruido_paso3()
    return varianza_ruido(img, mascara_marcas)


# ---------------------------------------------------------------------------
# Filtro de Lee (ventana 7x7).
# Con la media local m y la varianza local v de cada ventana:
#     k = max(0, (v − var_ruido) / v)        salida = m + k · (píxel − m)
# - Si la ventana varía lo mismo que el ruido (v ≈ var_ruido), k ≈ 0 y la
#   salida es la media: suaviza mucho, porque esa variación es solo speckle.
# - Si la ventana varía mucho más que el ruido (un borde), k ≈ 1 y la salida
#   es el píxel original: casi no toca los bordes.
# peso_lee devuelve el k de cada píxel (lo usan las figuras del paso 4).
# filtro_lee devuelve una imagen float (sin redondear), del mismo tamaño que la entrada.
# ---------------------------------------------------------------------------
def peso_lee(img, var_ruido):
    _, v = media_y_varianza_local(img)
    k = np.zeros_like(v)
    hay_variacion = v > 0
    k[hay_variacion] = np.maximum(0, (v[hay_variacion] - var_ruido) / v[hay_variacion])
    return k


def filtro_lee(img, var_ruido):
    m, _ = media_y_varianza_local(img)
    k = peso_lee(img, var_ruido)
    return m + k * (img.astype(float) - m)