# paso4_filtros.py
# Comparar filtros para suavizar el speckle: cuánto suavizan el tejido, cuánto
# conservan el borde de la lesión y cuánto afectan la textura.
# Se usan las 647 imágenes con lesión (benignas y malignas), limpias de marcas.
# Se corre desde la carpeta del proyecto:  python3 paso4_filtros.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso4/.

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import wilcoxon, mannwhitneyu
from skimage.feature import graycomatrix, graycoprops

from comun import armar_tabla, cargar
from marcas import limpiar
from filtros import (filtro_media, filtro_mediana, filtro_gaussiano, filtro_butterworth,
                     filtro_gaussiano_frecuencial, filtro_lee, varianza_ruido, abanico)

CARPETA_SALIDA = "outputs/paso4"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

df = armar_tabla()
df = df[df["clase"] != "normal"].reset_index(drop=True)   # 647 imágenes con lesión

# Parches de tejido elegidos en el paso 3 (solo los de imágenes con lesión)
parches = pd.read_csv("outputs/paso3/parches.csv")
parches = parches[parches["id"].isin(df["id"])]

TAM_PARCHE = 32
NOMBRES = ["sin filtrar", "media", "mediana", "gaussiano", "butterworth", "gaussiano frec.", "lee"]


def titulo(texto):
    """Imprime un título visible en la terminal."""
    print()
    print("=" * 70)
    print(texto)
    print("=" * 70)


def numero(x, decimales=3):
    """Escribe un número con coma decimal (estilo español), sin ceros de más al final."""
    texto = f"{x:.{decimales}f}"
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return texto.replace(".", ",")


def formato_p(p):
    """Escribe el p-valor sin exponentes: 'p < 0,001' o 'p = 0,042'."""
    if p < 0.001:
        return "p < 0,001"
    return "p = " + numero(p, 3)


def aplicar_filtros(limpia, mascara_marcas):
    """Devuelve un diccionario {nombre: imagen filtrada} con la imagen sin filtrar y los 6 filtros."""
    return {
        "sin filtrar": limpia.astype(float),
        "media": filtro_media(limpia),
        "mediana": filtro_mediana(limpia),
        "gaussiano": filtro_gaussiano(limpia),
        "butterworth": filtro_butterworth(limpia),
        "gaussiano frec.": filtro_gaussiano_frecuencial(limpia),
        "lee": filtro_lee(limpia, varianza_ruido(limpia, mascara_marcas)),
    }


def propiedades_glcm(imagen, region):
    """Contraste y energía de la GLCM dentro de una región (promedio de los 4 ángulos).
    32 niveles fijos: nivel = gris // 8 (0 a 31). Los píxeles fuera de la región
    se ponen en un nivel extra (32) y después se descartan de la matriz."""
    niveles = np.clip(np.floor(imagen / 8), 0, 31).astype(np.uint8)
    niveles[~region] = 32
    angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    P = graycomatrix(niveles, distances=[1], angles=angulos, levels=33, symmetric=True)
    P = P[:32, :32, :, :]   # solo pares de píxeles con los dos dentro de la región
    return graycoprops(P, "contrast").mean(), graycoprops(P, "energy").mean()


# ---------------------------------------------------------------------------
# BLOQUE 1. Los 6 filtros sobre una imagen de ejemplo.
# Todos están emparejados para suavizar lo mismo que un gaussiano de σ = 2
# (la explicación de cómo se emparejan está en filtros.py):
#   - media y mediana: ventana 7x7, porque raíz(12σ² + 1) = 7;
#   - gaussiano espacial: σ = 2;
#   - Butterworth (orden 2) y gaussiano frecuencial: D0 = 1/(2πσ) ≈ 0,08 ciclos/píxel,
#     con padding por reflexión antes de la FFT;
#   - Lee: ventana 7x7 y var_ruido = mediana de la varianza local de la imagen.
# Mostramos un recorte alrededor de la lesión para que se vean las diferencias.
# ---------------------------------------------------------------------------
titulo("BLOQUE 1. Filtros sobre una imagen de ejemplo")

ID_EJEMPLO = "benign (209)"
img, lesion = cargar(df[df["id"] == ID_EJEMPLO].iloc[0])
limpia, mascara_marcas = limpiar(img)
versiones = aplicar_filtros(limpia, mascara_marcas)

filas_lesion, columnas_lesion = np.where(lesion)
y0, y1 = max(filas_lesion.min() - 60, 0), filas_lesion.max() + 60
x0, x1 = max(columnas_lesion.min() - 60, 0), columnas_lesion.max() + 60

fig, ejes = plt.subplots(2, 4, figsize=(16, 8))
for eje, nombre in zip(ejes.ravel(), NOMBRES):
    eje.imshow(versiones[nombre][y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
    eje.set_title(nombre)
    eje.axis("off")
ejes.ravel()[-1].axis("off")
plt.suptitle(f"{ID_EJEMPLO} (recorte alrededor de la lesión)")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "01_filtros_ejemplo.png")
plt.savefig(ruta_figura, dpi=110)
plt.close()
print(f"Figura guardada en {ruta_figura}")


# ---------------------------------------------------------------------------
# Recorrido de las 647 imágenes: se calculan todas las medidas de los
# bloques 2 y 3 de una sola vez, para filtrar cada imagen una sola vez.
#
# Medidas (bloque 3):
# a) Ganancia de ENL. ENL ("número equivalente de looks") = (media / desvío)²
#    en un parche de tejido homogéneo: cuanto más suavizado, menos desvío y
#    más ENL. Ganancia = ENL filtrado / ENL original, en los parches del paso 3.
#    Ojo: la teoría del ENL es para la intensidad sin comprimir; sobre imágenes
#    con compresión logarítmica el valor absoluto no tiene sentido físico, así
#    que solo sirve para comparar filtros entre sí.
# b) Contraste de borde = gradiente de Sobel medio en una banda de 3 px sobre
#    el contorno de la lesión ÷ gradiente medio en un anillo de tejido de 5 a
#    20 px por fuera del borde (dentro del abanico). Más alto = el borde
#    resalta más sobre el "ruido" del tejido de alrededor.
# c) Textura: GLCM (matriz de co-ocurrencia de grises) con 32 niveles fijos,
#    distancia 1 y 4 ángulos (0°, 45°, 90°, 135°), en la lesión y en el anillo.
#    De la GLCM sacamos:
#      - contraste = cuánto cambia el gris entre vecinos (alto = textura áspera);
#      - energía = cuán repetidos son los pares de grises (alto = textura uniforme).
#    Con los 647 valores de lesión y los 647 de tejido calculamos el AUC:
#    la probabilidad de que la propiedad ordene bien una lesión y un tejido
#    elegidos al azar. Reportamos max(AUC, 1 − AUC), porque nos importa que
#    separe, no hacia qué lado (0,5 = no separa nada; 1 = separa perfecto).
#
# Control (bloque 2): diferencia media entre el gaussiano espacial y el
# frecuencial, lejos de los bordes de la imagen (a más de 20 px).
# ---------------------------------------------------------------------------
titulo("Procesando las 647 imágenes (puede tardar unos minutos)")

MARGEN_CONTROL = 20
diferencias_gauss = []
ganancias_enl = {nombre: [] for nombre in NOMBRES}
contraste_borde = {nombre: [] for nombre in NOMBRES}
glcm_lesion = {nombre: [] for nombre in NOMBRES}     # (contraste, energía) por imagen
glcm_tejido = {nombre: [] for nombre in NOMBRES}

for numero_imagen, (_, fila) in enumerate(df.iterrows(), start=1):
    img, lesion = cargar(fila)
    limpia, mascara_marcas = limpiar(img)
    versiones = aplicar_filtros(limpia, mascara_marcas)

    # Control: gaussiano espacial vs. frecuencial, lejos de los bordes
    m = MARGEN_CONTROL
    diferencia = versiones["gaussiano"][m:-m, m:-m] - versiones["gaussiano frec."][m:-m, m:-m]
    diferencias_gauss.append(np.abs(diferencia).mean())

    # Banda sobre el contorno y anillo de tejido alrededor
    contorno = lesion & ~ndimage.binary_erosion(lesion)
    banda = ndimage.binary_dilation(contorno, np.ones((3, 3)))
    distancia = ndimage.distance_transform_edt(~lesion)
    anillo = (distancia >= 5) & (distancia <= 20) & abanico(limpia)

    # Recorte alrededor de lesión + anillo, para que la GLCM sea más rápida
    filas_reg, columnas_reg = np.where(lesion | anillo)
    recorte = (slice(filas_reg.min(), filas_reg.max() + 1), slice(columnas_reg.min(), columnas_reg.max() + 1))

    parches_imagen = parches[parches["id"] == fila["id"]]

    for nombre in NOMBRES:
        version = versiones[nombre]

        # a) ENL en los parches de tejido
        for _, parche in parches_imagen.iterrows():
            y, x = parche["y"], parche["x"]
            original = limpia[y:y + TAM_PARCHE, x:x + TAM_PARCHE].astype(float)
            filtrado = version[y:y + TAM_PARCHE, x:x + TAM_PARCHE]
            enl_original = (original.mean() / original.std()) ** 2
            enl_filtrado = (filtrado.mean() / filtrado.std()) ** 2
            ganancias_enl[nombre].append(enl_filtrado / enl_original)

        # b) Contraste de borde
        gradiente = np.hypot(ndimage.sobel(version, axis=0), ndimage.sobel(version, axis=1))
        contraste_borde[nombre].append(gradiente[banda].mean() / gradiente[anillo].mean())

        # c) Textura (GLCM) en la lesión y en el anillo
        glcm_lesion[nombre].append(propiedades_glcm(version[recorte], lesion[recorte]))
        glcm_tejido[nombre].append(propiedades_glcm(version[recorte], anillo[recorte]))

    if numero_imagen % 100 == 0:
        print(f"  {numero_imagen} de {len(df)}")


# ---------------------------------------------------------------------------
# BLOQUE 2. Control: gaussiano espacial vs. gaussiano frecuencial.
# Si el emparejamiento D0 = 1/(2πσ) está bien, los dos tienen que dar
# prácticamente la misma imagen.
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Control: gaussiano espacial vs. frecuencial")

diferencias_gauss = np.array(diferencias_gauss)
print(f"Diferencia absoluta media (a más de {MARGEN_CONTROL} px del borde), promedio de las "
      f"{len(diferencias_gauss)} imágenes: {numero(diferencias_gauss.mean(), 4)} niveles de gris")
print(f"Máximo entre imágenes: {numero(diferencias_gauss.max(), 4)} niveles de gris")


# ---------------------------------------------------------------------------
# BLOQUE 3. Medidas por filtro (ENL, contraste de borde y AUC de textura).
# Las definiciones están en el comentario del recorrido, más arriba.
# ---------------------------------------------------------------------------
titulo("BLOQUE 3. Medidas por filtro")

print(f"a) Parches de tejido usados para el ENL: {len(parches)}")
print(f"b) y c) Imágenes con lesión: {len(df)}")

auc = {}
for nombre in NOMBRES:
    lesion_vals = np.array(glcm_lesion[nombre])
    tejido_vals = np.array(glcm_tejido[nombre])
    auc[nombre] = []
    for columna in [0, 1]:    # 0 = contraste, 1 = energía
        u, _ = mannwhitneyu(lesion_vals[:, columna], tejido_vals[:, columna])
        valor = u / (len(lesion_vals) * len(tejido_vals))
        auc[nombre].append(max(valor, 1 - valor))


# ---------------------------------------------------------------------------
# BLOQUE 4. Tabla final y comparación del contraste de borde.
# Una fila por filtro con la mediana de la ganancia de ENL, la mediana del
# contraste de borde y el AUC de textura (contraste y energía de la GLCM).
# Después comparamos el contraste de borde de la mediana y de Lee (los dos
# filtros pensados para cuidar los bordes) contra todos los demás, con
# Wilcoxon pareado (mismas imágenes).
# ---------------------------------------------------------------------------
titulo("BLOQUE 4. Tabla final")

tabla = pd.DataFrame({
    "filtro": NOMBRES,
    "ganancia ENL (mediana)": [np.median(ganancias_enl[n]) for n in NOMBRES],
    "contraste borde (mediana)": [np.median(contraste_borde[n]) for n in NOMBRES],
    "AUC contraste GLCM": [auc[n][0] for n in NOMBRES],
    "AUC energía GLCM": [auc[n][1] for n in NOMBRES],
})
print(tabla.to_string(index=False, formatters={c: (lambda v: numero(v, 3)) for c in tabla.columns[1:]}))

print("\nContraste de borde: Wilcoxon pareado (mismas imágenes)")
for referencia in ["mediana", "lee"]:
    print(f"\n  {referencia} contra los demás:")
    for otro in NOMBRES:
        if otro == referencia:
            continue
        a = np.array(contraste_borde[referencia])
        b = np.array(contraste_borde[otro])
        _, p = wilcoxon(a, b)
        print(f"    vs. {otro:16s} n = {len(a)}   mediana {referencia} = {numero(np.median(a), 3)}   "
              f"mediana {otro} = {numero(np.median(b), 3)}   {formato_p(p)}")


# ---------------------------------------------------------------------------
# DECISIÓN
# - Para segmentar: el filtro con mayor contraste de borde (mediana).
# - Para la textura: si el AUC sin filtrar es igual o mayor que el del mejor
#   filtro, conviene medirla sin filtrar (filtrar borra la textura que queremos medir).
# ---------------------------------------------------------------------------
titulo("DECISIÓN")

solo_filtros = tabla[tabla["filtro"] != "sin filtrar"]
mejor_borde = solo_filtros.loc[solo_filtros["contraste borde (mediana)"].idxmax()]
borde_sin_filtrar = tabla.loc[tabla["filtro"] == "sin filtrar", "contraste borde (mediana)"].iloc[0]
print(f"- Para segmentar: {mejor_borde['filtro']} "
      f"(contraste de borde mediano {numero(mejor_borde['contraste borde (mediana)'], 3)}; "
      f"sin filtrar: {numero(borde_sin_filtrar, 3)}).")

for columna, propiedad in [("AUC contraste GLCM", "contraste"), ("AUC energía GLCM", "energía")]:
    sin_filtrar = tabla.loc[tabla["filtro"] == "sin filtrar", columna].iloc[0]
    mejor = solo_filtros.loc[solo_filtros[columna].idxmax()]
    if sin_filtrar >= mejor[columna]:
        conclusion = "conviene medirla sin filtrar"
    else:
        conclusion = f"filtrar con {mejor['filtro']} separa mejor"
    print(f"- Textura ({propiedad} de la GLCM): AUC sin filtrar = {numero(sin_filtrar, 3)}, "
          f"mejor filtro = {mejor['filtro']} ({numero(mejor[columna], 3)}) -> {conclusion}.")
