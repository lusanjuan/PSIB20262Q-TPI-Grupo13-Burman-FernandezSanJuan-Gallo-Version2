# paso6_clasificador.py
# Clasificar las lesiones en benignas y malignas con 5 descriptores y una
# regresión logística. Los descriptores se calculan dos veces:
#   A) con la máscara manual (lo mejor que se puede lograr con estos descriptores);
#   B) con nuestra segmentación del paso 5 (lo que pasaría en un uso real).
# Se usan las imágenes con lesión (benignas y malignas), limpias de marcas.
# Se corre desde la carpeta del proyecto:  python3 paso6_clasificador.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso6_clasificador/.

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from comun import armar_tabla, cargar
from marcas import limpiar
from descriptores import huella, mancha_del_clic, descriptores, iou_elipse

CARPETA_SALIDA = "outputs/paso6_clasificador"
CARPETA_SEGMENTACIONES = "outputs/paso5_segmentacion/segmentaciones"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

df = armar_tabla()
df = df[df["clase"] != "normal"].reset_index(drop=True)   # 647 imágenes con lesión
df["maligna"] = (df["clase"] == "malignant").astype(int)

DESCRIPTORES = ["circularidad", "alto/ancho", "ecogenicidad", "sombra", "homogeneidad"]
SEMILLA = 13


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


def sensibilidad_especificidad(y, probabilidad, umbral):
    """Sensibilidad = malignas detectadas / malignas; especificidad = benignas bien dadas / benignas."""
    prediccion = probabilidad >= umbral
    sensibilidad = (prediccion & (y == 1)).sum() / (y == 1).sum()
    especificidad = (~prediccion & (y == 0)).sum() / (y == 0).sum()
    return sensibilidad, especificidad


def clasificador():
    """Regresión logística con estandarización (cada descriptor con media 0 y
    desvío 1, para que ninguno pese más por su escala) y class_weight="balanced"
    (las malignas son menos, así que cada una pesa más en el ajuste)."""
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))


# ---------------------------------------------------------------------------
# Recorrido de las 647 imágenes: se calcula todo de una vez.
# - Clic: el mismo que en el paso 5 (el punto más adentro de la máscara manual).
# - Huella para los duplicados (bloque 1).
# - Descriptores A (máscara manual) y B (segmentación del paso 5), en los dos
#   casos con la mancha que contiene el clic (bloque 2).
# - IoU de la máscara manual con su elipse equivalente (bloque 5).
# ---------------------------------------------------------------------------
titulo("Procesando las 647 imágenes (tarda unos minutos)")

huellas = []
filas_A, filas_B, ious = [], [], []
segmentacion_vacia = []
for numero_imagen, (_, fila) in enumerate(df.iterrows(), start=1):
    img, lesion = cargar(fila)
    limpia, _ = limpiar(img)
    huellas.append(huella(limpia).ravel())

    clic = np.unravel_index(np.argmax(ndimage.distance_transform_edt(lesion)), lesion.shape)
    manual = mancha_del_clic(lesion, clic)
    nuestra = np.array(Image.open(os.path.join(CARPETA_SEGMENTACIONES, fila["id"] + ".png"))) > 0
    nuestra = mancha_del_clic(nuestra, clic)

    filas_A.append(descriptores(limpia, manual))
    ious.append(iou_elipse(manual))
    segmentacion_vacia.append(not nuestra.any())
    filas_B.append(descriptores(limpia, nuestra) if nuestra.any() else {d: np.nan for d in DESCRIPTORES})

    if numero_imagen % 100 == 0:
        print(f"  {numero_imagen} de {len(df)}", flush=True)

A = pd.DataFrame(filas_A)
B = pd.DataFrame(filas_B)
df["segmentacion vacia"] = segmentacion_vacia
df["iou elipse"] = ious


# ---------------------------------------------------------------------------
# BLOQUE 1. Duplicados.
# En el paso 1 vimos imágenes casi iguales (la misma lesión, otra toma). Si
# una queda en entrenamiento y otra en prueba, el clasificador "ya la vio" y
# el resultado sale optimista. Por eso las agrupamos y nunca las separamos.
# - Huella de cada imagen (descriptores.py): imagen de 64x64 menos su versión
#   suavizada con un gaussiano σ = 4.
# - Correlación de Pearson entre las huellas de todos los pares de imágenes.
# - Dos imágenes son duplicadas si r >= 0,4, y los grupos se unen en cadena:
#   si A se parece a B y B a C, las tres van al mismo grupo.
# ---------------------------------------------------------------------------
titulo("BLOQUE 1. Duplicados")

UMBRAL_DUPLICADO = 0.4
H = np.array(huellas)
H = (H - H.mean(axis=1, keepdims=True)) / H.std(axis=1, keepdims=True)
correlacion = H @ H.T / H.shape[1]           # Pearson entre todos los pares
duplicadas = correlacion >= UMBRAL_DUPLICADO
np.fill_diagonal(duplicadas, False)
_, grupos = connected_components(duplicadas, directed=False)
df["grupo"] = grupos

# Justificación del umbral: si dos imágenes son la misma lesión, tienen que
# ser de la misma clase. Para cada rango de r miramos qué % de los pares es
# de la misma clase y lo comparamos con el % esperado por azar (dos imágenes
# cualesquiera elegidas al azar). Donde el % se despega del azar y se acerca
# al 100 %, los pares parecidos son en su mayoría duplicados reales.
i, j = np.triu_indices(len(df), k=1)            # todos los pares, cada uno una vez
r_pares = correlacion[i, j]
clases = df["clase"].to_numpy()
misma_clase = clases[i] == clases[j]
n_ben, n_mal = (clases == "benign").sum(), (clases == "malignant").sum()
n_total = len(clases)
azar = (n_ben * (n_ben - 1) + n_mal * (n_mal - 1)) / (n_total * (n_total - 1))

print(f"% de pares de la misma clase según r (esperado por azar: {numero(100 * azar, 1)} %)")
print(f"  {'rango de r':12s} {'pares':>8s} {'% misma clase':>15s}")
limites = [-np.inf, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, np.inf]
nombres_rangos = ["< 0,2", "0,2–0,3", "0,3–0,4", "0,4–0,5", "0,5–0,7", "0,7–0,9", ">= 0,9"]
for desde, hasta, nombre_rango in zip(limites[:-1], limites[1:], nombres_rangos):
    en_rango = (r_pares >= desde) & (r_pares < hasta)
    porcentaje = numero(100 * misma_clase[en_rango].mean(), 1) + " %" if en_rango.any() else "-"
    print(f"  {nombre_rango:12s} {en_rango.sum():>8d} {porcentaje:>15s}")
print()

tamanio_grupo = df["grupo"].map(df["grupo"].value_counts())
print(f"Imágenes con al menos un duplicado (r >= {numero(UMBRAL_DUPLICADO, 1)}): {(tamanio_grupo > 1).sum()}")
print(f"Grupos de duplicados (2 o más imágenes): {df.loc[tamanio_grupo > 1, 'grupo'].nunique()}")
print(f"Grupo más grande: {tamanio_grupo.max()} imágenes")


# ---------------------------------------------------------------------------
# BLOQUE 2. Los 5 descriptores (explicados en descriptores.py).
# Sobre la imagen limpia SIN filtrar, con A) la máscara manual y B) nuestra
# segmentación del paso 5. Solo imágenes con segmentación no vacía (si nuestra
# segmentación quedó vacía, B no tiene descriptores) y con todos los
# descriptores calculables. Mostramos la mediana por clase.
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Descriptores (mediana por clase)")

sin_nan = A[DESCRIPTORES].notna().all(axis=1) & B[DESCRIPTORES].notna().all(axis=1)
usar = ~df["segmentacion vacia"] & sin_nan
print(f"Excluidas por segmentación vacía: {df['segmentacion vacia'].sum()}")
print(f"Excluidas por algún descriptor no calculable (sombra fuera de la imagen): "
      f"{(~df['segmentacion vacia'] & ~sin_nan).sum()}")
print(f"Imágenes usadas: {usar.sum()}")

datos = df[usar].reset_index(drop=True)
A = A[usar].reset_index(drop=True)
B = B[usar].reset_index(drop=True)
y = datos["maligna"].to_numpy()
grupos = datos["grupo"].to_numpy()

for nombre, tabla in [("A (máscara manual)", A), ("B (nuestra segmentación)", B)]:
    print(f"\n{nombre}")
    medianas = tabla.groupby(datos["clase"]).median().T
    print(medianas.to_string(formatters={c: (lambda v: numero(v, 3)) for c in medianas.columns}))


# ---------------------------------------------------------------------------
# BLOQUE 3. Mirar los descriptores (las 646 imágenes).
# - AUC benigna/maligna de cada descriptor solo: la probabilidad de que una
#   maligna al azar tenga un valor más alto que una benigna al azar.
#   0,5 = no separa; cerca de 1 = más alto en malignas; cerca de 0 = más bajo
#   en malignas (separa igual, hacia el otro lado).
# - Correlación de Spearman entre los 5 (si dos están muy correlacionados,
#   miden casi lo mismo).
# - Boxplots por clase (figura 03_boxplots.png).
# ---------------------------------------------------------------------------
titulo("BLOQUE 3. AUC de cada descriptor y correlación")

auc_descriptores = pd.DataFrame({
    "A (manual)": [roc_auc_score(y, A[d]) for d in DESCRIPTORES],
    "B (nuestra)": [roc_auc_score(y, B[d]) for d in DESCRIPTORES],
}, index=DESCRIPTORES)
print(f"AUC benigna/maligna (n = {len(y)} imágenes: {(y == 0).sum()} benignas, {(y == 1).sum()} malignas):")
print(auc_descriptores.to_string(formatters={c: (lambda v: numero(v, 3)) for c in auc_descriptores.columns}))

for nombre, tabla in [("A (máscara manual)", A), ("B (nuestra segmentación)", B)]:
    print(f"\nCorrelación de Spearman entre descriptores, {nombre}:")
    rho = tabla.corr(method="spearman")
    print(rho.to_string(formatters={c: (lambda v: numero(v, 2)) for c in rho.columns}))

fig, ejes = plt.subplots(2, 5, figsize=(18, 8))
for i, (nombre, tabla) in enumerate([("A (manual)", A), ("B (nuestra)", B)]):
    for j, d in enumerate(DESCRIPTORES):
        valores = [tabla.loc[datos["clase"] == c, d] for c in ["benign", "malignant"]]
        ejes[i, j].boxplot(valores)
        ejes[i, j].set_xticks([1, 2])
        ejes[i, j].set_xticklabels(["benignas", "malignas"])
        ejes[i, j].set_title(f"{d} - {nombre}\nAUC = {numero(auc_descriptores.loc[d, nombre], 3)}", fontsize=10)
plt.suptitle(f"Descriptores por clase (n = {len(y)})")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "03_boxplots.png")
plt.savefig(ruta_figura, dpi=110)
plt.close()
print(f"\nFigura guardada en {ruta_figura}")


# ---------------------------------------------------------------------------
# BLOQUE 4. Clasificador: regresión logística (estandarización + class_weight
# "balanced"), con los 5 descriptores, para A y para B.
# Validación cruzada sobre las 646 imágenes: StratifiedGroupKFold de 5
# pliegues (semilla 13), con los grupos de duplicados (un grupo queda entero
# en un solo pliegue) y la misma proporción de clases en cada pliegue. Los
# mismos pliegues para A y para B. En cada pliegue:
#   - se entrena con los otros 4 y se evalúa en el que queda (cada imagen se
#     evalúa una sola vez, con un modelo que no la vio);
#   - umbral de sensibilidad: se elige SOLO con los 4 pliegues de entrenamiento.
#     Con ellos se hace otra validación cruzada interna (5 pliegues, también
#     agrupada), cada imagen recibe una probabilidad de un modelo que no la vio,
#     y se toma el umbral más alto que detecta al menos el 90 % de las malignas
#     (sensibilidad >= 0,90). Ese umbral se aplica al pliegue de prueba.
#     En diagnóstico se prefiere no dejar pasar malignas aunque haya más
#     falsas alarmas.
# Se reporta, para A y B, la media ± desvío de los 5 pliegues de:
#   - AUC;
#   - sensibilidad y especificidad con umbral 0,5 y con el umbral de sensibilidad;
#   - coeficiente de cada descriptor (estandarizado: más lejos de 0 = pesa más;
#     positivo = empuja hacia maligna; negativo = hacia benigna).
# Curva ROC de A y B con las predicciones de los 5 pliegues juntas
# (figura 04_roc_validacion.png).
# ---------------------------------------------------------------------------
titulo("BLOQUE 4. Clasificador (regresión logística, validación cruzada de 5 pliegues)")

validacion = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
pliegues = list(validacion.split(A, y, groups=grupos))     # los mismos para A y B
print("Imágenes de prueba en cada pliegue (benignas / malignas): "
      + "   ".join(f"{(y[i_pru] == 0).sum()} / {(y[i_pru] == 1).sum()}" for _, i_pru in pliegues))


def umbral_sensibilidad_90(X, y_ent, grupos_ent):
    """Umbral más alto con sensibilidad >= 0,90, con las probabilidades de una
    validación cruzada interna agrupada de 5 pliegues (solo datos de entrenamiento)."""
    probabilidad_afuera = np.zeros(len(y_ent))
    for i_ent, i_val in validacion.split(X, y_ent, groups=grupos_ent):
        modelo = clasificador().fit(X[i_ent], y_ent[i_ent])
        probabilidad_afuera[i_val] = modelo.predict_proba(X[i_val])[:, 1]
    _, sensibilidades, umbrales = roc_curve(y_ent, probabilidad_afuera)
    return umbrales[sensibilidades >= 0.90].max()


def media_desvio(valores, decimales=3):
    return f"{numero(np.mean(valores), decimales)} ± {numero(np.std(valores), decimales)}"


probabilidades = {}
for nombre, tabla in [("A (manual)", A), ("B (nuestra)", B)]:
    X = tabla[DESCRIPTORES].to_numpy()
    probabilidad = np.zeros(len(y))
    auc_pliegues, umbrales_pliegues, coeficientes = [], [], []
    sens_05, espec_05, sens_90, espec_90 = [], [], [], []

    for i_ent, i_pru in pliegues:
        umbral = umbral_sensibilidad_90(X[i_ent], y[i_ent], grupos[i_ent])
        modelo = clasificador().fit(X[i_ent], y[i_ent])
        probabilidad[i_pru] = modelo.predict_proba(X[i_pru])[:, 1]

        auc_pliegues.append(roc_auc_score(y[i_pru], probabilidad[i_pru]))
        umbrales_pliegues.append(umbral)
        coeficientes.append(modelo[-1].coef_[0])
        s, e = sensibilidad_especificidad(y[i_pru], probabilidad[i_pru], 0.5)
        sens_05.append(s)
        espec_05.append(e)
        s, e = sensibilidad_especificidad(y[i_pru], probabilidad[i_pru], umbral)
        sens_90.append(s)
        espec_90.append(e)

    probabilidades[nombre] = probabilidad
    coeficientes = np.array(coeficientes)

    print(f"\n{nombre} (n = {len(y)}, media ± desvío de los 5 pliegues)")
    print(f"  AUC = {media_desvio(auc_pliegues)}")
    print(f"  Umbral 0,5:                    sensibilidad = {media_desvio(sens_05)}   "
          f"especificidad = {media_desvio(espec_05)}")
    print(f"  Umbral de sensibilidad >= 0,9: sensibilidad = {media_desvio(sens_90)}   "
          f"especificidad = {media_desvio(espec_90)}")
    print(f"     (umbral elegido en cada pliegue: {', '.join(numero(u, 3) for u in umbrales_pliegues)})")
    print("  Coeficientes (descriptores estandarizados; + = hacia maligna, − = hacia benigna):")
    for k, d in enumerate(DESCRIPTORES):
        print(f"     {d:14s} {media_desvio(coeficientes[:, k], 2)}")

fig, eje = plt.subplots(figsize=(6.5, 6))
for nombre, color in [("A (manual)", "tab:green"), ("B (nuestra)", "tab:orange")]:
    fpr, tpr, _ = roc_curve(y, probabilidades[nombre])
    eje.plot(fpr, tpr, color=color, linewidth=2,
             label=f"{nombre}: AUC (5 pliegues juntos) = {numero(roc_auc_score(y, probabilidades[nombre]), 3)}")
eje.plot([0, 1], [0, 1], color="gray", linestyle="--", label="azar")
eje.set_xlabel("1 − especificidad")
eje.set_ylabel("Sensibilidad")
eje.set_title(f"Curvas ROC, validación cruzada (n = {len(y)})")
eje.legend(loc="lower right")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "04_roc_validacion.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"\nFigura guardada en {ruta_figura}")

# ---------------------------------------------------------------------------
# BLOQUE 5. Sesgo de las máscaras manuales.
# Si quien dibujó las máscaras trazó las benignas como óvalos lisos y las
# malignas con más detalle, la circularidad de A estaría midiendo en parte el
# dibujo y no la lesión. Lo medimos con el IoU de cada máscara manual (la
# mancha del clic) con su elipse equivalente: cerca de 1 = casi una elipse.
# Comparamos benignas y malignas con Mann-Whitney (todas las 647 imágenes).
# ---------------------------------------------------------------------------
titulo("BLOQUE 5. Sesgo de las máscaras: IoU con la elipse equivalente")

benignas = df.loc[df["clase"] == "benign", "iou elipse"]
malignas = df.loc[df["clase"] == "malignant", "iou elipse"]
_, p = mannwhitneyu(benignas, malignas)
print("Test: Mann-Whitney U (dos colas)")
print(f"  Benignas: n = {len(benignas)}, IoU mediano = {numero(benignas.median(), 3)}")
print(f"  Malignas: n = {len(malignas)}, IoU mediano = {numero(malignas.median(), 3)}")
print(f"  {formato_p(p)}")
