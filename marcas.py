# marcas.py
# Funciones para detectar y borrar las marcas del ecógrafo
# (cruces de medición, líneas punteadas, números y textos).
# Se importa así:  from marcas import detectar_marcas, rellenar, limpiar

import numpy as np
import cv2
from skimage.morphology import disk, opening, dilation
from skimage.measure import label, regionprops
from skimage.restoration import inpaint_biharmonic

# Parámetros (los justifica paso2_marcas.py)
RADIO_DISCO = 3          # disco de 7 px de diámetro; las marcas miden ~3 px de ancho
UMBRAL_TOPHAT = 100      # un píxel es candidato a marca si sobresale >= 100 de su entorno
FIBRA_ELONGACION = 3     # largo / ancho
FIBRA_LARGO = 8          # píxeles
FIBRA_INCLINACION = 35   # grados respecto de la horizontal
AGRANDAR = 2             # píxeles que se agranda la máscara final
METODO = "biharmonico"   # método de relleno que ganó en paso2_marcas.py (bloque 4)


# ---------------------------------------------------------------------------
# detectar_marcas(img): devuelve una máscara True/False con las marcas.
#
# Idea: top-hat blanco = imagen − apertura.
# La apertura es una erosión (mínimo en el disco) seguida de una dilatación
# (máximo en el disco). Todo lo claro MÁS ANGOSTO que el disco desaparece.
#
# Ejemplo en 1D, con una "ventana" de 3 píxeles (el disco en 1D):
#     imagen:     80  80 220  80  80      <- una marca de 1 px de ancho
#     erosión:    80  80  80  80  80      (mínimo de cada ventana: la marca no
#                                          entra entera en ninguna, se pierde)
#     dilatación: 80  80  80  80  80      (máximo de cada ventana: ya no queda
#                                          nada claro que recuperar)
#     apertura =  80  80  80  80  80
#     top-hat  =   0   0 140   0   0      <- solo queda la marca
#
# En cambio, algo claro MÁS ANCHO que la ventana sobrevive:
#     imagen:     80 150 150 150 150  80
#     erosión:    80  80 150 150  80  80  (las ventanas del medio caben enteras)
#     dilatación: 80 150 150 150 150  80  (el máximo lo vuelve a ensanchar)
#     top-hat  =   0   0   0   0   0   0  <- el tejido ancho no aparece
#
# Después:
#   - umbral: top-hat >= 100;
#   - se descartan las fibras del tejido (claras y finas, pero largas y casi
#     horizontales): elongación >= 3, largo >= 8 px e inclinación <= 35°;
#   - se agranda la máscara 2 px para tapar el borde suave de la marca.
# Con devolver_fibras=True devuelve también las fibras descartadas (para las
# figuras del paso 2).
# ---------------------------------------------------------------------------
def detectar_marcas(img, devolver_fibras=False):
    apertura = opening(img, disk(RADIO_DISCO))
    tophat = img.astype(int) - apertura.astype(int)
    candidatos = tophat >= UMBRAL_TOPHAT

    etiquetas = label(candidatos)
    son_marca, son_fibra = [], []
    for objeto in regionprops(etiquetas):
        largo = objeto.axis_major_length
        ancho = max(objeto.axis_minor_length, 1)
        elongacion = largo / ancho
        # orientation = ángulo respecto del eje vertical; lo pasamos a "respecto de la horizontal"
        inclinacion = 90 - abs(np.degrees(objeto.orientation))
        es_fibra = (elongacion >= FIBRA_ELONGACION and largo >= FIBRA_LARGO
                    and inclinacion <= FIBRA_INCLINACION)
        (son_fibra if es_fibra else son_marca).append(objeto.label)

    # np.isin marca todos los objetos de una vez (más rápido que uno por uno)
    marcas = dilation(np.isin(etiquetas, son_marca), disk(AGRANDAR))
    if devolver_fibras:
        return marcas, np.isin(etiquetas, son_fibra)
    return marcas


# ---------------------------------------------------------------------------
# rellenar(img, mascara): rellena los píxeles de la máscara con el método que
# ganó la validación del paso 2 (bloque 4).
# limpiar(img): detecta las marcas y las rellena. Devuelve (imagen_limpia, mascara).
# ---------------------------------------------------------------------------
def rellenar(img, mascara):
    if not mascara.any():
        return img.copy()
    if METODO == "telea":
        return cv2.inpaint(img, mascara.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    relleno = inpaint_biharmonic(img / 255.0, mascara)
    return np.clip(np.round(relleno * 255), 0, 255).astype(np.uint8)


def limpiar(img):
    mascara = detectar_marcas(img)
    return rellenar(img, mascara), mascara