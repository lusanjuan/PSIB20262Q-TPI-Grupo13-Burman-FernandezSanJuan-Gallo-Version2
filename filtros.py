# filtros.py
# Filtros para suavizar el speckle, todos con el mismo nivel de suavizado
# (equivalente a un gaussiano con σ = 2 px), para poder compararlos en forma justa.
# Se importa así:  from filtros import filtro_media, filtro_lee, ...
#
# Cómo se emparejan con el gaussiano de σ = 2:
#   - Media y mediana: ventana cuadrada de L x L. Un promedio uniforme de L
#     píxeles tiene varianza (L² − 1) / 12; igualándola a σ² queda
#     L = raíz(12σ² + 1) = raíz(49) = 7. Ventana de 7 x 7.
#   - Gaussiano espacial: σ = 2 directamente.
#   - Butterworth y gaussiano frecuencial: la transformada de Fourier de un
#     gaussiano de ancho σ es otro gaussiano de ancho 1 / (2πσ). Por eso la
#     frecuencia de corte es D0 = 1 / (2π · 2) ≈ 0,08 ciclos/píxel.
#   - Lee: ventana de 7 x 7, igual que la media.
#
# Todos devuelven una imagen float (sin redondear), del mismo tamaño que la entrada.

import numpy as np
from scipy import ndimage

SIGMA = 2
VENTANA = 7                          # raíz(12 · 2² + 1) = 7
D0 = 1 / (2 * np.pi * SIGMA)         # ≈ 0,0796 ciclos/píxel
ORDEN_BUTTERWORTH = 2
PADDING = 64                         # píxeles de reflexión alrededor antes de la FFT


def filtro_media(img):
    """Promedio en una ventana de 7x7."""
    return ndimage.uniform_filter(img.astype(float), VENTANA)


def filtro_mediana(img):
    """Mediana en una ventana de 7x7 (no promedia: elige el valor del medio)."""
    return ndimage.median_filter(img, VENTANA).astype(float)


def filtro_gaussiano(img):
    """Gaussiano espacial con σ = 2."""
    return ndimage.gaussian_filter(img.astype(float), SIGMA)


# ---------------------------------------------------------------------------
# Filtros en frecuencia.
# Pasos: agrandar la imagen reflejándola en los bordes (padding), pasar a
# frecuencia con la FFT, multiplicar por H(D) (D = distancia al centro del
# espectro, en ciclos/píxel), volver con la FFT inversa y recortar el padding.
# El padding por reflexión evita que la FFT "pegue" el borde derecho con el
# izquierdo (la FFT supone que la imagen se repite), lo que crearía saltos
# falsos y artefactos en los bordes.
# ---------------------------------------------------------------------------
def _filtrar_en_frecuencia(img, H_de_D):
    x = np.pad(img.astype(float), PADDING, mode="symmetric")
    fy = np.fft.fftfreq(x.shape[0])
    fx = np.fft.fftfreq(x.shape[1])
    D = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    salida = np.real(np.fft.ifft2(np.fft.fft2(x) * H_de_D(D)))
    return salida[PADDING:-PADDING, PADDING:-PADDING]


def filtro_butterworth(img):
    """Butterworth pasabajos de orden 2: H = 1 / (1 + (D/D0)^(2·orden)).
    En D = D0 vale 1/2: D0 es la frecuencia donde deja pasar la mitad.
    Por debajo de D0 pasa casi todo y por encima corta, con una caída más
    abrupta que el gaussiano."""
    return _filtrar_en_frecuencia(img, lambda D: 1 / (1 + (D / D0) ** (2 * ORDEN_BUTTERWORTH)))


def filtro_gaussiano_frecuencial(img):
    """Gaussiano en frecuencia: H = exp(−D² / (2·D0²)).
    D0 es el ancho (desvío) del gaussiano en frecuencia. Con D0 = 1/(2πσ) da
    el mismo resultado que el gaussiano espacial de σ = 2."""
    return _filtrar_en_frecuencia(img, lambda D: np.exp(-D ** 2 / (2 * D0 ** 2)))


# ---------------------------------------------------------------------------
# Abanico de la sonda y varianza del ruido de cada imagen (los usa Lee).
# Abanico = lo que no es fondo negro (gris > 0), con los huecos rellenados
# (dentro del tejido oscuro hay píxeles sueltos en 0 que no son fondo).
# Varianza del ruido = mediana de la varianza local 7x7, solo en ventanas
# enteras dentro del abanico y sin marcas (igual que el bloque 5 del paso 3).
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
# ---------------------------------------------------------------------------
def filtro_lee(img, var_ruido):
    m, v = media_y_varianza_local(img)
    k = np.zeros_like(v)
    hay_variacion = v > 0
    k[hay_variacion] = np.maximum(0, (v[hay_variacion] - var_ruido) / v[hay_variacion])
    return m + k * (img.astype(float) - m)
