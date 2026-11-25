# paso5_segmentacion.py
# Segmentar la lesión a partir de un clic, con 3 métodos, y compararlos.
# Todos trabajan sobre la imagen filtrada con Lee (paso 4).
# Se usan las 647 imágenes con lesión (benignas y malignas), limpias de marcas.
# Se corre desde la carpeta del proyecto:  python3 paso5_segmentacion.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso5_segmentacion/.
#
# REGLA GENERAL: la máscara manual se usa SOLO para dos cosas:
#   1) simular el clic del usuario (un punto dentro de la lesión);
#   2) evaluar (comparar la segmentación con la lesión de referencia).
# Nunca se usa para guiar la segmentación ni para recortar: los métodos solo
# ven la imagen y el clic, igual que en un uso real.

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.stats import wilcoxon, mannwhitneyu

from comun import armar_tabla, cargar
from marcas import limpiar
from filtros import filtro_lee, varianza_ruido
from segmentacion import segmentar, ventana_alrededor, METODOS, USA_TOLERANCIA

CARPETA_SALIDA = "outputs/paso5_segmentacion"
CARPETA_SEGMENTACIONES = os.path.join(CARPETA_SALIDA, "segmentaciones")
os.makedirs(CARPETA_SEGMENTACIONES, exist_ok=True)

# Dice de todos los métodos y parámetros: si el archivo existe, se lee en vez de recalcular
RUTA_DICE = os.path.join(CARPETA_SALIDA, "dice_todos.csv")

df = armar_tabla()
df = df[df["clase"] != "normal"].reset_index(drop=True)   # 647 imágenes con lesión

VENTANAS = [150, 200, 300]
TOLERANCIAS = [15, 20, 25, 30, 35]


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


def dice(a, b):
    """Dice = 2·|A∩B| / (|A| + |B|). 1 = coinciden exacto, 0 = no se tocan."""
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return 2 * (a & b).sum() / total


# ---------------------------------------------------------------------------
# BLOQUE 1. Preparación.
# - Clic simulado: el punto de la máscara más alejado de su borde (el máximo de
#   la transformada de distancia). Es el punto "más adentro" de la lesión,
#   como haría alguien que marca el centro.
# - Lesión de referencia: la mancha de la máscara que contiene el clic (si la
#   imagen tiene varias lesiones, solo se evalúa la que se clickeó).
# - Separación en ajuste (30 %) y evaluación (70 %), estratificada por clase
#   (la misma proporción de benignas y malignas en cada parte), semilla 13.
#   Los parámetros se eligen SOLO con ajuste y los resultados se reportan SOLO
#   con evaluación: si se eligieran y evaluaran con las mismas imágenes, el
#   resultado saldría optimista.
# - La imagen entera se filtra con Lee (con la varianza del ruido de esa
#   imagen, como en el paso 4) y después todos los métodos trabajan dentro de
#   una ventana cuadrada centrada en el clic.
# ---------------------------------------------------------------------------
titulo("BLOQUE 1. Preparación")


def clic_y_referencia(lesion):
    distancia = ndimage.distance_transform_edt(lesion)
    clic = np.unravel_index(np.argmax(distancia), lesion.shape)
    etiquetas, _ = ndimage.label(lesion)
    referencia = etiquetas == etiquetas[clic]
    return clic, referencia


rng = np.random.default_rng(13)
df["subconjunto"] = "evaluación"
for clase in ["benign", "malignant"]:
    indices = df.index[df["clase"] == clase].to_numpy()
    indices = rng.permutation(indices)
    n_ajuste = round(0.3 * len(indices))
    df.loc[indices[:n_ajuste], "subconjunto"] = "ajuste"

print(pd.crosstab(df["clase"], df["subconjunto"], margins=True, margins_name="total").to_string())


# ---------------------------------------------------------------------------
# Cálculo del Dice de todos los métodos, ventanas y tolerancias, en las 647
# imágenes. Calcular todo de una vez es más rápido (cada imagen se filtra una
# sola vez); después, la elección de parámetros usa SOLO las filas de ajuste y
# la evaluación SOLO las filas de evaluación.
# Los métodos están explicados en segmentacion.py, todos sobre la imagen con Lee:
#   a) region growing (crecimiento de regiones);
#   b) Otsu (umbralización);
#   c) k-means con 2 grupos (agrupamiento).
# Todos terminan con el mismo refinamiento: rellenar agujeros, apertura con
# disco de radio 3 y quedarse con la mancha que contiene el clic.
# ---------------------------------------------------------------------------
titulo("Calculando las segmentaciones (647 imágenes; tarda unos minutos)")

if os.path.exists(RUTA_DICE):
    resultados = pd.read_csv(RUTA_DICE)
    print(f"Resultados leídos de {RUTA_DICE} (no se recalculan las segmentaciones)")
else:
    filas = []
    for numero_imagen, (_, fila) in enumerate(df.iterrows(), start=1):
        img, lesion = cargar(fila)
        limpia, mascara_marcas = limpiar(img)
        con_lee = filtro_lee(limpia, varianza_ruido(limpia, mascara_marcas))
        clic, referencia = clic_y_referencia(lesion)   # la máscara solo se usa acá...
        for lado in VENTANAS:
            ventana = ventana_alrededor(limpia.shape, clic, lado)
            clic_ventana = (clic[0] - ventana[0].start, clic[1] - ventana[1].start)
            for metodo in METODOS:
                tolerancias = TOLERANCIAS if metodo in USA_TOLERANCIA else [0]
                for tolerancia in tolerancias:
                    segmentacion = segmentar(metodo, con_lee[ventana], clic_ventana, tolerancia)
                    completa = np.zeros(limpia.shape, dtype=bool)
                    completa[ventana] = segmentacion
                    filas.append({
                        "id": fila["id"], "clase": fila["clase"], "subconjunto": fila["subconjunto"],
                        "metodo": metodo, "ventana": lado, "tolerancia": tolerancia,
                        "dice": dice(completa, referencia),          # ...y acá, para evaluar
                    })
        if numero_imagen % 50 == 0:
            print(f"  {numero_imagen} de {len(df)}", flush=True)
    resultados = pd.DataFrame(filas)
    resultados.to_csv(RUTA_DICE, index=False)
    print(f"Resultados guardados en {RUTA_DICE}")


# ---------------------------------------------------------------------------
# BLOQUE 2. Ajuste de parámetros (SOLO con el subconjunto de ajuste).
# Para cada método probamos ventanas de 150, 200 y 300 px y, en el region
# growing, tolerancias de 15 a 35 niveles de gris. Elegimos la combinación
# con mayor Dice medio en las imágenes de ajuste.
# (La tolerancia vale 0 en los métodos que no la usan.)
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Ajuste de parámetros (subconjunto de ajuste)")

ajuste = resultados[resultados["subconjunto"] == "ajuste"]
promedios = ajuste.groupby(["metodo", "ventana", "tolerancia"])["dice"].mean().reset_index()
elegidos = promedios.loc[promedios.groupby("metodo")["dice"].idxmax()].set_index("metodo")

print(f"Imágenes de ajuste: {ajuste['id'].nunique()}")
for metodo in METODOS:
    e = elegidos.loc[metodo]
    texto_tolerancia = f", tolerancia {int(e['tolerancia'])}" if metodo in USA_TOLERANCIA else ""
    print(f"  {metodo:18s} ventana {int(e['ventana'])} px{texto_tolerancia}   "
          f"(Dice medio en ajuste = {numero(e['dice'], 3)})")


# ---------------------------------------------------------------------------
# BLOQUE 3. Evaluación (SOLO con el subconjunto de evaluación).
# Dice = 2·|A∩B| / (|A| + |B|), con A = segmentación y B = lesión de
# referencia. 1 = coinciden exacto; 0 = no se tocan.
# Tabla con Dice medio y mediano, % de imágenes con Dice >= 0,8 (buena
# segmentación) y % con Dice < 0,2 (fracaso), ordenada de mayor a menor.
# Después, el mejor método contra cada uno de los demás con Wilcoxon pareado
# (las mismas imágenes con los dos métodos).
# ---------------------------------------------------------------------------
titulo("BLOQUE 3. Evaluación (subconjunto de evaluación)")

evaluacion = resultados[resultados["subconjunto"] == "evaluación"]
partes = []
for metodo in METODOS:
    e = elegidos.loc[metodo]
    partes.append(evaluacion[(evaluacion["metodo"] == metodo) & (evaluacion["ventana"] == e["ventana"])
                             & (evaluacion["tolerancia"] == e["tolerancia"])])
dice_eval = pd.concat(partes).pivot(index="id", columns="metodo", values="dice")
clase_eval = evaluacion.drop_duplicates("id").set_index("id").loc[dice_eval.index, "clase"]

tabla = pd.DataFrame({
    "método": METODOS,
    "Dice medio": [dice_eval[m].mean() for m in METODOS],
    "Dice mediano": [dice_eval[m].median() for m in METODOS],
    "% Dice >= 0,8": [100 * (dice_eval[m] >= 0.8).mean() for m in METODOS],
    "% Dice < 0,2": [100 * (dice_eval[m] < 0.2).mean() for m in METODOS],
}).sort_values("Dice medio", ascending=False).reset_index(drop=True)

print(f"Imágenes de evaluación: {len(dice_eval)}\n")
print(tabla.to_string(index=False, formatters={
    "Dice medio": lambda v: numero(v, 3), "Dice mediano": lambda v: numero(v, 3),
    "% Dice >= 0,8": lambda v: numero(v, 1) + " %", "% Dice < 0,2": lambda v: numero(v, 1) + " %"}))

orden = list(tabla["método"])
fig, eje = plt.subplots(figsize=(10, 5))
eje.violinplot([dice_eval[m] for m in orden], showmedians=True)
eje.set_xticks(range(1, len(orden) + 1))
eje.set_xticklabels(orden)
eje.set_ylabel("Dice")
eje.set_ylim(0, 1)
eje.set_title(f"Dice por método (evaluación, n = {len(dice_eval)} imágenes; línea = mediana)")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "03_violines_dice.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"\nFigura guardada en {ruta_figura}")

mejor = orden[0]
print(f"\nMejor método: {mejor}. Comparación con Wilcoxon pareado (mismas imágenes):")
for otro in orden[1:]:
    _, p = wilcoxon(dice_eval[mejor], dice_eval[otro])
    print(f"  vs. {otro:18s} n = {len(dice_eval)}   media {mejor} = {numero(dice_eval[mejor].mean(), 3)}   "
          f"media {otro} = {numero(dice_eval[otro].mean(), 3)}   {formato_p(p)}")


# ---------------------------------------------------------------------------
# BLOQUE 4. Benignas vs. malignas.
# ¿El mejor método segmenta igual de bien las dos clases? Mann-Whitney.
# ---------------------------------------------------------------------------
titulo("BLOQUE 4. Mejor método: benignas vs. malignas")

benignas = dice_eval.loc[clase_eval == "benign", mejor]
malignas = dice_eval.loc[clase_eval == "malignant", mejor]
_, p = mannwhitneyu(benignas, malignas)
print(f"Método: {mejor}. Test: Mann-Whitney U (dos colas)")
print(f"  Benignas: n = {len(benignas)}, Dice medio = {numero(benignas.mean(), 3)}")
print(f"  Malignas: n = {len(malignas)}, Dice medio = {numero(malignas.mean(), 3)}")
print(f"  {formato_p(p)}")


# ---------------------------------------------------------------------------
# BLOQUE 5. Mejores y peores segmentaciones, y guardado para el paso 6.
# Segmentamos las 647 imágenes con el mejor método y sus parámetros, y
# guardamos cada máscara como PNG (blanco = lesión) en outputs/paso5_segmentacion/segmentaciones/.
# Figura: las 3 mejores y las 3 peores del subconjunto de evaluación, con el
# contorno manual (verde) y el propio (rojo), mostrando la ventana del método.
# ---------------------------------------------------------------------------
titulo(f"BLOQUE 5. Segmentaciones del mejor método ({mejor})")

lado = int(elegidos.loc[mejor, "ventana"])
tolerancia = elegidos.loc[mejor, "tolerancia"]
para_figura = {}
ordenadas = dice_eval[mejor].sort_values()
ids_figura = list(ordenadas.index[-3:][::-1]) + list(ordenadas.index[:3])

for _, fila in df.iterrows():
    img, lesion = cargar(fila)
    limpia, mascara_marcas = limpiar(img)
    con_lee = filtro_lee(limpia, varianza_ruido(limpia, mascara_marcas))
    clic, referencia = clic_y_referencia(lesion)
    ventana = ventana_alrededor(limpia.shape, clic, lado)
    clic_ventana = (clic[0] - ventana[0].start, clic[1] - ventana[1].start)
    completa = np.zeros(limpia.shape, dtype=bool)
    completa[ventana] = segmentar(mejor, con_lee[ventana], clic_ventana, tolerancia)
    Image.fromarray((completa * 255).astype(np.uint8)).save(
        os.path.join(CARPETA_SEGMENTACIONES, fila["id"] + ".png"))
    if fila["id"] in ids_figura:
        para_figura[fila["id"]] = (limpia[ventana], referencia[ventana], completa[ventana])
print(f"{len(df)} segmentaciones guardadas en {CARPETA_SEGMENTACIONES}/")

fig, ejes = plt.subplots(2, 3, figsize=(14, 9.5))
for eje, id_imagen in zip(ejes.ravel(), ids_figura):
    imagen, referencia, propia = para_figura[id_imagen]
    eje.imshow(imagen, cmap="gray", vmin=0, vmax=255)
    eje.contour(referencia, levels=[0.5], colors="lime", linewidths=1.5)
    if propia.any():
        eje.contour(propia, levels=[0.5], colors="red", linewidths=1.5)
    eje.set_title(f"{id_imagen}: Dice = {numero(dice_eval.loc[id_imagen, mejor], 3)}", fontsize=10)
    eje.axis("off")
ejes[0, 0].set_ylabel("mejores")
ejes[1, 0].set_ylabel("peores")
plt.suptitle(f"{mejor}: 3 mejores (arriba) y 3 peores (abajo). Verde = manual, rojo = propia")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "05_mejores_y_peores.png")
plt.savefig(ruta_figura, dpi=110)
plt.close()
print(f"Figura guardada en {ruta_figura}")
