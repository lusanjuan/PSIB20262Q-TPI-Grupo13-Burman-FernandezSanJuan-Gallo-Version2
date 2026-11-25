# paso4_filtros.py
# Suavizar el speckle con el filtro de Lee y medir qué le hace a la imagen:
# cuánto suaviza el tejido, cuánto conserva el borde de la lesión y cuánto
# afecta la textura. Siempre comparado con la imagen sin filtrar.
# Se usan las 647 imágenes con lesión (benignas y malignas), limpias de marcas.
# Se corre desde la carpeta del proyecto:  python3 paso4_filtros.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso4_filtros/.

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
from filtros import filtro_lee, varianza_ruido, abanico

CARPETA_SALIDA = "outputs/paso4_filtros"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

df = armar_tabla()
df = df[df["clase"] != "normal"].reset_index(drop=True)   # 647 imágenes con lesión

# Parches de tejido elegidos en el paso 3 (solo los de imágenes con lesión)
parches = pd.read_csv("outputs/paso3_speckle/parches.csv")
parches = parches[parches["id"].isin(df["id"])]

TAM_PARCHE = 32
NOMBRES = ["sin filtrar", "lee"]
COLORES = {"sin filtrar": "tab:gray", "lee": "tab:orange"}


def titulo(texto):
    """Imprime un título visible en la terminal."""
    print()
    print("=" * 70)
    print(texto, flush=True)
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


def versiones_de(limpia, mascara_marcas):
    """Devuelve {nombre: imagen} con la imagen sin filtrar y la filtrada con Lee,
    y la varianza del ruido de la imagen que usó Lee."""
    var_ruido = varianza_ruido(limpia, mascara_marcas)
    return {"sin filtrar": limpia.astype(float), "lee": filtro_lee(limpia, var_ruido)}, var_ruido


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
# BLOQUE 1. Lee sobre una imagen de ejemplo.
# Lee (7x7) es el filtro adaptativo para speckle visto en clase (detalle en
# filtros.py): suaviza mucho donde la variación de la ventana se parece a la
# del ruido y casi nada donde es mucho mayor (los bordes).
# La varianza del ruido es un valor por imagen: la mediana de la varianza
# local 7x7, dentro del abanico y sin marcas (como en el bloque 5 del paso 3).
# Mostramos un recorte alrededor de la lesión, sin filtrar y con Lee.
# ---------------------------------------------------------------------------
titulo("BLOQUE 1. Lee sobre una imagen de ejemplo")

ID_EJEMPLO = "benign (209)"
img, lesion = cargar(df[df["id"] == ID_EJEMPLO].iloc[0])
limpia, mascara_marcas = limpiar(img)
versiones, var_ejemplo = versiones_de(limpia, mascara_marcas)
print(f"Varianza del ruido de {ID_EJEMPLO}: {numero(var_ejemplo, 1)}")

filas_lesion, columnas_lesion = np.where(lesion)
y0, y1 = max(filas_lesion.min() - 60, 0), filas_lesion.max() + 60
x0, x1 = max(columnas_lesion.min() - 60, 0), columnas_lesion.max() + 60

fig, ejes = plt.subplots(1, 2, figsize=(11, 5))
for eje, nombre in zip(ejes, NOMBRES):
    eje.imshow(versiones[nombre][y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
    eje.set_title(nombre)
    eje.axis("off")
plt.suptitle(f"{ID_EJEMPLO} (recorte alrededor de la lesión)")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "01_lee_ejemplo.png")
plt.savefig(ruta_figura, dpi=110)
plt.close()
print(f"Figura guardada en {ruta_figura}")


# ---------------------------------------------------------------------------
# Recorrido de las 647 imágenes: se calculan todas las medidas de una vez.
#
# a) Ganancia de ENL. ENL ("número equivalente de looks") = (media / desvío)²
#    en un parche de tejido homogéneo: cuanto más suavizado, menos desvío y
#    más ENL. Ganancia = ENL con Lee / ENL sin filtrar, en los parches del paso 3.
#    Ojo: la teoría del ENL es para la intensidad sin comprimir; sobre imágenes
#    con compresión logarítmica el valor absoluto no tiene sentido físico, así
#    que solo sirve para comparar con la imagen sin filtrar.
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
# ---------------------------------------------------------------------------
titulo("Procesando las 647 imágenes (puede tardar unos minutos)")

varianzas_ruido = []
ganancias_enl = []
contraste_borde = {nombre: [] for nombre in NOMBRES}
glcm_lesion = {nombre: [] for nombre in NOMBRES}     # (contraste, energía) por imagen
glcm_tejido = {nombre: [] for nombre in NOMBRES}

for numero_imagen, (_, fila) in enumerate(df.iterrows(), start=1):
    img, lesion = cargar(fila)
    limpia, mascara_marcas = limpiar(img)
    versiones, var_ruido = versiones_de(limpia, mascara_marcas)
    varianzas_ruido.append(var_ruido)

    # Banda sobre el contorno y anillo de tejido alrededor
    contorno = lesion & ~ndimage.binary_erosion(lesion)
    banda = ndimage.binary_dilation(contorno, np.ones((3, 3)))
    distancia = ndimage.distance_transform_edt(~lesion)
    anillo = (distancia >= 5) & (distancia <= 20) & abanico(limpia)

    # Recorte alrededor de lesión + anillo, para que la GLCM sea más rápida
    filas_reg, columnas_reg = np.where(lesion | anillo)
    recorte = (slice(filas_reg.min(), filas_reg.max() + 1), slice(columnas_reg.min(), columnas_reg.max() + 1))

    # a) ENL en los parches de tejido de esta imagen
    for _, parche in parches[parches["id"] == fila["id"]].iterrows():
        zona = (slice(parche["y"], parche["y"] + TAM_PARCHE), slice(parche["x"], parche["x"] + TAM_PARCHE))
        original = versiones["sin filtrar"][zona]
        filtrado = versiones["lee"][zona]
        enl_original = (original.mean() / original.std()) ** 2
        enl_filtrado = (filtrado.mean() / filtrado.std()) ** 2
        ganancias_enl.append(enl_filtrado / enl_original)

    for nombre in NOMBRES:
        version = versiones[nombre]

        # b) Contraste de borde
        gradiente = np.hypot(ndimage.sobel(version, axis=0), ndimage.sobel(version, axis=1))
        contraste_borde[nombre].append(gradiente[banda].mean() / gradiente[anillo].mean())

        # c) Textura (GLCM) en la lesión y en el anillo
        glcm_lesion[nombre].append(propiedades_glcm(version[recorte], lesion[recorte]))
        glcm_tejido[nombre].append(propiedades_glcm(version[recorte], anillo[recorte]))

    if numero_imagen % 100 == 0:
        print(f"  {numero_imagen} de {len(df)}", flush=True)

varianzas_ruido = np.array(varianzas_ruido)
ganancias_enl = np.array(ganancias_enl)

print(f"\nVarianza del ruido usada por Lee (un valor por imagen, n = {len(varianzas_ruido)}):")
print(f"  mínimo {numero(varianzas_ruido.min(), 1)}   percentil 25 {numero(np.percentile(varianzas_ruido, 25), 1)}   "
      f"mediana {numero(np.median(varianzas_ruido), 1)}   percentil 75 {numero(np.percentile(varianzas_ruido, 75), 1)}   "
      f"máximo {numero(varianzas_ruido.max(), 1)}")


# ---------------------------------------------------------------------------
# BLOQUE 2. Resultados: Lee vs. sin filtrar.
# a) Ganancia de ENL en los parches de tejido (mediana; 1 = sin cambio).
# b) Contraste de borde, con Wilcoxon pareado (las mismas imágenes con y sin Lee).
# c) AUC de textura lesión vs. tejido, con y sin Lee.
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Lee vs. sin filtrar")

print(f"a) Ganancia de ENL en {len(ganancias_enl)} parches de tejido: mediana = {numero(np.median(ganancias_enl), 2)}  "
      f"(percentil 25 = {numero(np.percentile(ganancias_enl, 25), 2)}, "
      f"percentil 75 = {numero(np.percentile(ganancias_enl, 75), 2)})")

sin_filtrar = np.array(contraste_borde["sin filtrar"])
con_lee = np.array(contraste_borde["lee"])
_, p = wilcoxon(con_lee, sin_filtrar)
print(f"b) Contraste de borde (n = {len(con_lee)} imágenes, Wilcoxon pareado): "
      f"mediana sin filtrar = {numero(np.median(sin_filtrar), 3)}   mediana con Lee = {numero(np.median(con_lee), 3)}   "
      f"{formato_p(p)}")
print(f"   Imágenes en que Lee aumenta el contraste de borde: {numero(100 * (con_lee > sin_filtrar).mean(), 1)} %")

auc = {}
for nombre in NOMBRES:
    lesion_vals = np.array(glcm_lesion[nombre])
    tejido_vals = np.array(glcm_tejido[nombre])
    auc[nombre] = []
    for columna in [0, 1]:    # 0 = contraste, 1 = energía
        u, _ = mannwhitneyu(lesion_vals[:, columna], tejido_vals[:, columna])
        valor = u / (len(lesion_vals) * len(tejido_vals))
        auc[nombre].append(max(valor, 1 - valor))

print("c) AUC de textura, lesión vs. tejido de alrededor (n = 647 lesiones y 647 anillos):")
for k, propiedad in enumerate(["contraste", "energía"]):
    print(f"   {propiedad:10s} de la GLCM: sin filtrar = {numero(auc['sin filtrar'][k], 3)}   "
          f"con Lee = {numero(auc['lee'][k], 3)}")


# ---------------------------------------------------------------------------
# BLOQUE 3. Qué le hace Lee a la imagen (figura 03_efecto_lee.png).
# Imagen con un borde de lesión nítido. Tomamos la fila que pasa por el punto
# más interior de la lesión y el borde izquierdo de la lesión en esa fila.
#   - Arriba: el mismo recorte sin filtrar y con Lee, con zoom sobre ese
#     pedazo de borde. La línea punteada roja es la del perfil.
#   - Abajo a la izquierda: perfil de grises a lo largo de esa línea, con y sin
#     Lee, y una línea vertical donde está el borde según la máscara. Muestra
#     cuánto aplana Lee el granito y qué le hace al salto del borde.
#   - Abajo a la derecha: distribución de la ganancia de ENL en los parches
#     de tejido (la línea en 1 = sin cambio).
# ---------------------------------------------------------------------------
titulo("BLOQUE 3. Efecto de Lee (figura)")

ID_EFECTO = "benign (38)"
MEDIO_ZOOM = 40   # recorte de 81 x 81 px
img, lesion = cargar(df[df["id"] == ID_EFECTO].iloc[0])
limpia, mascara_marcas = limpiar(img)
versiones, _ = versiones_de(limpia, mascara_marcas)

distancia_interior = ndimage.distance_transform_edt(lesion)
fila_perfil = np.unravel_index(np.argmax(distancia_interior), lesion.shape)[0]
columna_borde = np.where(lesion[fila_perfil])[0].min()     # borde izquierdo en esa fila
zoom = (slice(fila_perfil - MEDIO_ZOOM, fila_perfil + MEDIO_ZOOM + 1),
        slice(columna_borde - MEDIO_ZOOM, columna_borde + MEDIO_ZOOM + 1))

fig = plt.figure(figsize=(16, 11))
grilla = fig.add_gridspec(2, 4, height_ratios=[1, 0.9])
for j, nombre in enumerate(NOMBRES):
    eje = fig.add_subplot(grilla[0, 2 * j:2 * j + 2])
    eje.imshow(versiones[nombre][zoom], cmap="gray", vmin=0, vmax=255)
    eje.contour(lesion[zoom], levels=[0.5], colors="lime", linewidths=1)
    eje.axhline(MEDIO_ZOOM, color="red", linestyle="--", linewidth=1)
    eje.set_title(nombre, color=COLORES[nombre], fontsize=13)
    eje.axis("off")

eje = fig.add_subplot(grilla[1, :3])
x = np.arange(-MEDIO_ZOOM, MEDIO_ZOOM + 1)
for nombre in NOMBRES:
    eje.plot(x, versiones[nombre][zoom][MEDIO_ZOOM], color=COLORES[nombre],
             linewidth=1 if nombre == "sin filtrar" else 2, label=nombre)
eje.axvline(0, color="lime", linestyle=":", linewidth=2, label="borde (máscara)")
eje.set_xlabel("Distancia al borde (px; negativo = tejido, positivo = lesión)")
eje.set_ylabel("Nivel de gris")
eje.set_title("Perfil de grises a lo largo de la línea roja")
eje.legend()

eje = fig.add_subplot(grilla[1, 3])
eje.hist(ganancias_enl, bins=40, color=COLORES["lee"])
eje.axvline(1, color="black", linestyle="--", linewidth=1, label="1 = sin cambio")
eje.axvline(np.median(ganancias_enl), color="tab:red", linewidth=2,
            label=f"mediana = {numero(np.median(ganancias_enl), 2)}")
eje.set_xlabel("Ganancia de ENL (Lee / sin filtrar)")
eje.set_ylabel("Cantidad de parches")
eje.set_title("Cuánto ruido saca Lee en el tejido")
eje.legend()

plt.suptitle(f"{ID_EFECTO}: zoom sobre el borde de la lesión (verde = contorno manual)", fontsize=14)
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "03_efecto_lee.png")
plt.savefig(ruta_figura, dpi=110)
plt.close()
print(f"Figura guardada en {ruta_figura}")

# Números del perfil: granito = desvío del perfil en el tejido (antes del borde),
# salto = gris medio en la lesión menos gris medio en el tejido, a 10–30 px del borde
perfil_tejido = slice(MEDIO_ZOOM - 30, MEDIO_ZOOM - 10)
perfil_lesion = slice(MEDIO_ZOOM + 10, MEDIO_ZOOM + 30)
for nombre in NOMBRES:
    perfil = versiones[nombre][zoom][MEDIO_ZOOM]
    print(f"  {nombre:12s} desvío del perfil en el tejido = {numero(perfil[perfil_tejido].std(), 1):>5s}   "
          f"salto del borde = {numero(perfil[perfil_lesion].mean() - perfil[perfil_tejido].mean(), 1)}")


# ---------------------------------------------------------------------------
# CONCLUSIÓN
# - Para segmentar (paso 5) se usa la imagen filtrada con Lee.
# - Para la textura: si el AUC sin filtrar es igual o mayor que con Lee,
#   conviene medirla sin filtrar (filtrar borra la textura que queremos medir).
# ---------------------------------------------------------------------------
titulo("CONCLUSIÓN")

print("- Para segmentar (paso 5): imagen filtrada con Lee.")
for k, propiedad in enumerate(["contraste", "energía"]):
    if auc["sin filtrar"][k] >= auc["lee"][k]:
        conclusion = "conviene medirla sin filtrar"
    else:
        conclusion = "con Lee separa mejor"
    print(f"- Textura ({propiedad} de la GLCM): AUC sin filtrar = {numero(auc['sin filtrar'][k], 3)}, "
          f"con Lee = {numero(auc['lee'][k], 3)} -> {conclusion}.")
