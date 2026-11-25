# paso5_segmentacion.py
# Segmentar la lesión a partir de un clic, con 3 métodos, y compararlos.
# Todos trabajan sobre la imagen filtrada con Lee (paso 4).
# Se usan las imágenes con lesión (benignas y malignas) válidas del paso 1,
# limpias de marcas (las guardó el paso 2).
# Se corre desde la carpeta del proyecto:  python3 paso5_segmentacion.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso5_segmentacion/.
#
# REGLA GENERAL: la máscara manual se usa SOLO para dos cosas:
#   1) simular el clic del usuario (un punto dentro de la lesión);
#   2) evaluar (comparar la segmentación con la lesión de referencia).
# Nunca se usa para guiar la segmentación ni para recortar: los métodos solo
# ven la imagen y el clic, igual que en un uso real.
#
# Figuras:
#   01_clic_y_ventana.png        clic simulado, lesión de referencia y ventanas, en un ejemplo
#   02_ajuste_parametros.png     Dice medio en ajuste según ventana (y tolerancia del region growing)
#   03_violines_dice.png         Dice de cada método en evaluación
#   03b_metodos_ejemplo.png      los 3 métodos sobre una misma imagen
#   04a_dice_por_clase.png       Dice del mejor método en benignas y malignas
#   04b_dice_vs_tamanio.png      Dice del mejor método según el tamaño de la lesión
#   05_mejores_y_peores.png      las 3 mejores y las 3 peores segmentaciones
# Tablas:
#   dice_todos.csv               Dice de cada imagen, método y parámetros
#   tabla_evaluacion.csv         resumen por método en evaluación
#   parametros_elegidos.csv      el mejor método y sus parámetros (los usa la interfaz)

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.stats import wilcoxon, mannwhitneyu

from comun import armar_tabla, agregar_grupos, cargar_limpia
from filtros import filtro_lee, ruido_para_lee
from segmentacion import segmentar, ventana_alrededor, METODOS, USA_TOLERANCIA

CARPETA_SALIDA = "outputs/paso5_segmentacion"
CARPETA_SEGMENTACIONES = os.path.join(CARPETA_SALIDA, "segmentaciones")
os.makedirs(CARPETA_SEGMENTACIONES, exist_ok=True)

# Dice de todos los métodos y parámetros: si el archivo existe y corresponde a
# las mismas imágenes, se lee en vez de recalcular. Si cambia algo del
# procesamiento (filtro, varianza del ruido, métodos), poner RECALCULAR = True.
RUTA_DICE = os.path.join(CARPETA_SALIDA, "dice_todos.csv")
RECALCULAR = False

df = agregar_grupos(armar_tabla())
df = df[df["clase"] != "normal"].reset_index(drop=True)   # imágenes con lesión
N = len(df)

VENTANAS = [150, 200, 300]
TOLERANCIAS = [15, 20, 25, 30, 35]
COLORES_METODO = ["#2a78b5", "#e0a030", "#2e9e4f", "#c8553d", "#8c8c8c"]


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


def guardar_figura(nombre, dpi=120):
    """Guarda la figura actual en la carpeta de salida, la cierra e imprime la ruta."""
    ruta = os.path.join(CARPETA_SALIDA, nombre)
    plt.savefig(ruta, dpi=dpi)
    plt.close()
    print(f"Figura guardada en {ruta}")


def dice(a, b):
    """Dice = 2·|A∩B| / (|A| + |B|). 1 = coinciden exacto, 0 = no se tocan."""
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return 2 * (a & b).sum() / total


def preparar(fila):
    """Imagen limpia, imagen con Lee y lesión de una fila (Lee con la varianza
    del ruido que indique filtros.py, igual que en el paso 4)."""
    limpia, lesion, mascara_marcas = cargar_limpia(fila, con_marcas=True)
    con_lee = filtro_lee(limpia, ruido_para_lee(limpia, mascara_marcas))
    return limpia, con_lee, lesion


def segmentar_completa(metodo, con_lee, clic, lado, tolerancia):
    """Segmenta dentro de la ventana centrada en el clic y devuelve la máscara
    del tamaño de la imagen entera, y la ventana usada."""
    ventana = ventana_alrededor(con_lee.shape, clic, lado)
    clic_ventana = (clic[0] - ventana[0].start, clic[1] - ventana[1].start)
    completa = np.zeros(con_lee.shape, dtype=bool)
    completa[ventana] = segmentar(metodo, con_lee[ventana], clic_ventana, tolerancia)
    return completa, ventana


# ---------------------------------------------------------------------------
# BLOQUE 1. Preparación.
# - Clic simulado: el punto de la máscara más alejado de su borde (el máximo de
#   la transformada de distancia). Es el punto "más adentro" de la lesión,
#   como haría alguien que marca el centro.
# - Lesión de referencia: la mancha de la máscara que contiene el clic (si la
#   imagen tiene varias lesiones, solo se evalúa la que se clickeó).
# - Separación en ajuste (30 %) y evaluación (70 %), estratificada por clase
#   (la misma proporción de benignas y malignas en cada parte), semilla 13.
#   Se reparten GRUPOS de imágenes repetidas (paso 1), no imágenes sueltas:
#   si dos copias de la misma ecografía cayeran una en ajuste y otra en
#   evaluación, los parámetros se elegirían con una imagen que después se evalúa.
#   Los parámetros se eligen SOLO con ajuste y los resultados se reportan SOLO
#   con evaluación: si se eligieran y evaluaran con las mismas imágenes, el
#   resultado saldría optimista.
# - La imagen entera se filtra con Lee (como en el paso 4) y después todos los
#   métodos trabajan dentro de una ventana cuadrada centrada en el clic.
# Figura (01): en una imagen de ejemplo, el clic, la lesión de referencia y
# las tres ventanas que se prueban.
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
    grupos = rng.permutation(df.loc[df["clase"] == clase, "grupo"].unique())
    meta = round(0.3 * (df["clase"] == clase).sum())    # 30 % de las imágenes de la clase
    en_ajuste = 0
    for grupo in grupos:
        if en_ajuste >= meta:
            break
        filas_grupo = df.index[df["grupo"] == grupo]
        df.loc[filas_grupo, "subconjunto"] = "ajuste"
        en_ajuste += len(filas_grupo)

print(pd.crosstab(df["clase"], df["subconjunto"], margins=True, margins_name="total").to_string())
grupos_en_ambos = df.groupby("grupo")["subconjunto"].nunique().gt(1).sum()
print(f"Grupos de repetidas repartidos entre ajuste y evaluación: {grupos_en_ambos} (tiene que ser 0)")

# 01: clic y ventanas sobre un ejemplo
ID_EJEMPLO = "benign (209)" if "benign (209)" in set(df["id"]) else df["id"].iloc[0]
limpia, con_lee, lesion = preparar(df[df["id"] == ID_EJEMPLO].iloc[0])
clic, referencia = clic_y_referencia(lesion)
fig, eje = plt.subplots(figsize=(9, 7))
eje.imshow(con_lee, cmap="gray", vmin=0, vmax=255)
eje.contour(referencia, levels=[0.5], colors="lime", linewidths=1.5)
for lado, color in zip(VENTANAS, ["#ffd23f", "#ff8c42", "#ff3c38"]):
    v = ventana_alrededor(con_lee.shape, clic, lado)
    eje.add_patch(plt.Rectangle((v[1].start, v[0].start), v[1].stop - v[1].start, v[0].stop - v[0].start,
                                fill=False, edgecolor=color, linewidth=2, label=f"ventana {lado} px"))
eje.plot(clic[1], clic[0], "+", color="cyan", markersize=18, markeredgewidth=3, label="clic simulado")
eje.legend(loc="lower right")
eje.set_title(f"{ID_EJEMPLO} con Lee: clic (punto más interior), referencia (verde) y ventanas")
eje.axis("off")
plt.tight_layout()
guardar_figura("01_clic_y_ventana.png", dpi=100)


# ---------------------------------------------------------------------------
# Cálculo del Dice de todos los métodos, ventanas y tolerancias, en todas las
# imágenes con lesión. Calcular todo de una vez es más rápido (cada imagen se
# filtra una sola vez); después, la elección de parámetros usa SOLO las filas
# de ajuste y la evaluación SOLO las filas de evaluación.
# Los métodos están explicados en segmentacion.py, todos sobre la imagen con Lee:
#   a) region growing (crecimiento de regiones);
#   b) Otsu (umbralización);
#   c) k-means con 2 grupos (agrupamiento).
# Todos terminan con el mismo refinamiento: rellenar agujeros, apertura con
# disco de radio 3 y quedarse con la mancha que contiene el clic.
# Se guarda también el área de la lesión de referencia (para la figura 04b).
# ---------------------------------------------------------------------------
titulo(f"Calculando las segmentaciones ({N} imágenes; tarda unos minutos)")

resultados = None
if os.path.exists(RUTA_DICE) and not RECALCULAR:
    guardados = pd.read_csv(RUTA_DICE)
    if set(guardados["id"]) == set(df["id"]) and "area" in guardados.columns:
        resultados = guardados.drop(columns="subconjunto", errors="ignore")
        print(f"Resultados leídos de {RUTA_DICE} (no se recalculan las segmentaciones)")
    else:
        print(f"{RUTA_DICE} es de otras imágenes o de una versión anterior: se recalcula")

if resultados is None:
    filas = []
    for numero_imagen, (_, fila) in enumerate(df.iterrows(), start=1):
        limpia, con_lee, lesion = preparar(fila)
        clic, referencia = clic_y_referencia(lesion)   # la máscara solo se usa acá...
        for lado in VENTANAS:
            for metodo in METODOS:
                tolerancias = TOLERANCIAS if metodo in USA_TOLERANCIA else [0]
                for tolerancia in tolerancias:
                    completa, _ = segmentar_completa(metodo, con_lee, clic, lado, tolerancia)
                    filas.append({
                        "id": fila["id"], "clase": fila["clase"],
                        "metodo": metodo, "ventana": lado, "tolerancia": tolerancia,
                        "dice": dice(completa, referencia),          # ...y acá, para evaluar
                        "area": int(referencia.sum()),
                    })
        if numero_imagen % 50 == 0:
            print(f"  {numero_imagen} de {N}", flush=True)
    resultados = pd.DataFrame(filas)
    resultados.to_csv(RUTA_DICE, index=False)
    print(f"Resultados guardados en {RUTA_DICE}")

# El subconjunto se toma siempre de la separación de arriba (no del archivo)
resultados = resultados.merge(df[["id", "subconjunto"]], on="id")


# ---------------------------------------------------------------------------
# BLOQUE 2. Ajuste de parámetros (SOLO con el subconjunto de ajuste).
# Para cada método probamos ventanas de 150, 200 y 300 px y, en el region
# growing, tolerancias de 15 a 35 niveles de gris.
# Regla de elección: entre las combinaciones cuyo Dice medio en ajuste queda a
# menos de 0,01 del mejor (empatadas en la práctica), elegimos la que tiene
# MENOS fracasos (% de imágenes con Dice < 0,4). Un fracaso suele ser una fuga:
# la región se derrama al tejido de alrededor. Así, entre dos combinaciones que
# en promedio andan igual, se prefiere la más robusta (por ejemplo, una ventana
# más chica, que frena las fugas).
# (La tolerancia vale 0 en los métodos que no la usan.)
# Figura (02): Dice medio en ajuste según la ventana, para cada método; para
# los que usan tolerancia, una línea por tolerancia. El punto elegido va marcado.
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Ajuste de parámetros (subconjunto de ajuste)")

EMPATE = 0.01          # diferencias de Dice medio menores a esto se consideran empate
UMBRAL_FRACASO = 0.4   # Dice < 0,4 = fracaso (casi siempre, una fuga)

ajuste = resultados[resultados["subconjunto"] == "ajuste"]
promedios = (ajuste.groupby(["metodo", "ventana", "tolerancia"])["dice"]
             .agg(dice="mean", fracasos=lambda v: 100 * (v < UMBRAL_FRACASO).mean())
             .reset_index())
filas_elegidas = []
for metodo in METODOS:
    del_metodo = promedios[promedios["metodo"] == metodo]
    candidatos = del_metodo[del_metodo["dice"] >= del_metodo["dice"].max() - EMPATE]
    filas_elegidas.append(candidatos.sort_values(["fracasos", "dice"], ascending=[True, False]).iloc[0])
elegidos = pd.DataFrame(filas_elegidas).set_index("metodo")

print(f"Imágenes de ajuste: {ajuste['id'].nunique()}")
for metodo in METODOS:
    del_metodo = promedios[promedios["metodo"] == metodo]
    candidatos = del_metodo[del_metodo["dice"] >= del_metodo["dice"].max() - EMPATE]
    print(f"  {metodo}: combinaciones empatadas (Dice medio a menos de {numero(EMPATE, 2)} del mejor):")
    for _, c in candidatos.sort_values("dice", ascending=False).iterrows():
        texto_tolerancia = f", tolerancia {int(c['tolerancia'])}" if metodo in USA_TOLERANCIA else ""
        print(f"     ventana {int(c['ventana'])} px{texto_tolerancia}: Dice medio = {numero(c['dice'], 3)}, "
              f"fracasos (Dice < {numero(UMBRAL_FRACASO, 1)}) = {numero(c['fracasos'], 1)} %")
    e = elegidos.loc[metodo]
    texto_tolerancia = f", tolerancia {int(e['tolerancia'])}" if metodo in USA_TOLERANCIA else ""
    print(f"     -> elegida: ventana {int(e['ventana'])} px{texto_tolerancia} (la de menos fracasos)")

fig, ejes = plt.subplots(1, len(METODOS), figsize=(5 * len(METODOS), 4.5), sharey=True, squeeze=False)
for eje, metodo in zip(ejes[0], METODOS):
    del_metodo = promedios[promedios["metodo"] == metodo]
    tolerancias = sorted(del_metodo["tolerancia"].unique())
    mapa_color = plt.cm.viridis(np.linspace(0.1, 0.9, len(tolerancias)))
    for tolerancia, color in zip(tolerancias, mapa_color):
        linea = del_metodo[del_metodo["tolerancia"] == tolerancia].sort_values("ventana")
        etiqueta = f"tolerancia {int(tolerancia)}" if metodo in USA_TOLERANCIA else None
        eje.plot(linea["ventana"], linea["dice"], "o-", color=color if metodo in USA_TOLERANCIA else "black",
                 label=etiqueta)
    e = elegidos.loc[metodo]
    eje.plot(e["ventana"], e["dice"], "*", color="red", markersize=18, label="elegido")
    eje.set_xticks(VENTANAS)
    eje.set_xlabel("Ventana (px)")
    eje.set_title(metodo)
    eje.legend(fontsize=8)
    eje.spines[["top", "right"]].set_visible(False)
ejes[0, 0].set_ylabel("Dice medio (ajuste)")
plt.suptitle(f"Elección de parámetros con las {ajuste['id'].nunique()} imágenes de ajuste")
plt.tight_layout()
guardar_figura("02_ajuste_parametros.png")


# ---------------------------------------------------------------------------
# BLOQUE 3. Evaluación (SOLO con el subconjunto de evaluación).
# Dice = 2·|A∩B| / (|A| + |B|), con A = segmentación y B = lesión de
# referencia. 1 = coinciden exacto; 0 = no se tocan.
# Tabla con Dice medio y mediano, % de imágenes con Dice >= 0,8 (buena
# segmentación) y % con Dice < 0,2 (fracaso), ordenada de mayor a menor.
# Después, el mejor método contra cada uno de los demás con Wilcoxon pareado
# (las mismas imágenes con los dos métodos).
# Figura extra (03b): los métodos sobre una misma imagen de evaluación, la
# de Dice mediano del mejor método (un caso típico, ni el mejor ni el peor).
# ---------------------------------------------------------------------------
titulo("BLOQUE 3. Evaluación (subconjunto de evaluación)")

evaluacion = resultados[resultados["subconjunto"] == "evaluación"]
partes = []
for metodo in METODOS:
    e = elegidos.loc[metodo]
    partes.append(evaluacion[(evaluacion["metodo"] == metodo) & (evaluacion["ventana"] == e["ventana"])
                             & (evaluacion["tolerancia"] == e["tolerancia"])])
dice_eval = pd.concat(partes).pivot(index="id", columns="metodo", values="dice")
info_eval = evaluacion.drop_duplicates("id").set_index("id").loc[dice_eval.index]
clase_eval = info_eval["clase"]

tabla = pd.DataFrame({
    "método": METODOS,
    "Dice medio": [dice_eval[m].mean() for m in METODOS],
    "Dice mediano": [dice_eval[m].median() for m in METODOS],
    "% Dice >= 0,8": [100 * (dice_eval[m] >= 0.8).mean() for m in METODOS],
    "% Dice < 0,2": [100 * (dice_eval[m] < 0.2).mean() for m in METODOS],
}).sort_values("Dice medio", ascending=False).reset_index(drop=True)
tabla.to_csv(os.path.join(CARPETA_SALIDA, "tabla_evaluacion.csv"), index=False)

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
guardar_figura("03_violines_dice.png")

mejor = orden[0]
print(f"\nMejor método: {mejor}. Comparación con Wilcoxon pareado (mismas imágenes):")
for otro in orden[1:]:
    _, p = wilcoxon(dice_eval[mejor], dice_eval[otro])
    print(f"  vs. {otro:18s} n = {len(dice_eval)}   media {mejor} = {numero(dice_eval[mejor].mean(), 3)}   "
          f"media {otro} = {numero(dice_eval[otro].mean(), 3)}   {formato_p(p)}")

# 03b: los métodos sobre una imagen típica
ordenadas = dice_eval[mejor].sort_values()
id_tipica = ordenadas.index[len(ordenadas) // 2]
limpia, con_lee, lesion = preparar(df[df["id"] == id_tipica].iloc[0])
clic, referencia = clic_y_referencia(lesion)
lado_mayor = int(max(elegidos["ventana"]))
ventana_fig = ventana_alrededor(con_lee.shape, clic, lado_mayor)
fig, ejes = plt.subplots(1, len(METODOS), figsize=(5.5 * len(METODOS), 5.5), squeeze=False)
for eje, metodo in zip(ejes[0], METODOS):
    e = elegidos.loc[metodo]
    completa, _ = segmentar_completa(metodo, con_lee, clic, int(e["ventana"]), e["tolerancia"])
    eje.imshow(con_lee[ventana_fig], cmap="gray", vmin=0, vmax=255)
    eje.contour(referencia[ventana_fig], levels=[0.5], colors="lime", linewidths=1.5)
    if completa[ventana_fig].any():
        eje.contour(completa[ventana_fig], levels=[0.5], colors="red", linewidths=1.5)
    eje.set_title(f"{metodo}: Dice = {numero(dice(completa, referencia), 3)}")
    eje.axis("off")
plt.suptitle(f"{id_tipica} (Dice mediano de {mejor}). Verde = manual, rojo = método")
plt.tight_layout()
guardar_figura("03b_metodos_ejemplo.png", dpi=100)


# ---------------------------------------------------------------------------
# BLOQUE 4. Benignas vs. malignas.
# ¿El mejor método segmenta igual de bien las dos clases? Mann-Whitney.
# Figuras:
#   04a: Dice del mejor método en benignas y malignas.
#   04b: Dice del mejor método según el tamaño de la lesión (área en píxeles,
#        escala log). Muestra si los fracasos se concentran en lesiones chicas
#        o grandes.
# ---------------------------------------------------------------------------
titulo("BLOQUE 4. Mejor método: benignas vs. malignas")

benignas = dice_eval.loc[clase_eval == "benign", mejor]
malignas = dice_eval.loc[clase_eval == "malignant", mejor]
_, p = mannwhitneyu(benignas, malignas)
print(f"Método: {mejor}. Test: Mann-Whitney U (dos colas)")
print(f"  Benignas: n = {len(benignas)}, Dice medio = {numero(benignas.mean(), 3)}")
print(f"  Malignas: n = {len(malignas)}, Dice medio = {numero(malignas.mean(), 3)}")
print(f"  {formato_p(p)}")

COLORES_CLASE = {"benign": "#2a78b5", "malignant": "#c8553d"}
rng_puntos = np.random.default_rng(0)
fig, eje = plt.subplots(figsize=(6, 5))
valores = [benignas.to_numpy(), malignas.to_numpy()]
eje.boxplot(valores, widths=0.5, showfliers=False, medianprops={"color": "black", "linewidth": 2})
for k, (clase, v) in enumerate(zip(["benign", "malignant"], valores), start=1):
    eje.scatter(k + rng_puntos.uniform(-0.18, 0.18, len(v)), v, s=8, alpha=0.45,
                color=COLORES_CLASE[clase], edgecolors="none")
eje.set_xticks([1, 2], [f"benign\n(n = {len(benignas)})", f"malignant\n(n = {len(malignas)})"])
eje.set_ylim(0, 1)
eje.set_ylabel("Dice")
eje.set_title(f"{mejor}: Dice por clase (Mann-Whitney: {formato_p(p)})")
eje.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
guardar_figura("04a_dice_por_clase.png")

fig, eje = plt.subplots(figsize=(8, 5))
for clase in ["benign", "malignant"]:
    ids = clase_eval.index[clase_eval == clase]
    eje.scatter(info_eval.loc[ids, "area"], dice_eval.loc[ids, mejor], s=10, alpha=0.5,
                color=COLORES_CLASE[clase], edgecolors="none", label=clase)
eje.set_xscale("log")
eje.set_ylim(0, 1)
eje.set_xlabel("Área de la lesión de referencia (píxeles, escala log)")
eje.set_ylabel("Dice")
eje.set_title(f"{mejor}: Dice según el tamaño de la lesión (evaluación)")
eje.legend()
eje.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
guardar_figura("04b_dice_vs_tamanio.png")


# ---------------------------------------------------------------------------
# BLOQUE 5. Mejores y peores segmentaciones, y guardado para el paso 6.
# Segmentamos todas las imágenes con lesión con el mejor método y sus
# parámetros, y guardamos cada máscara como PNG (blanco = lesión) en
# outputs/paso5_segmentacion/segmentaciones/.
# Figura: las 3 mejores y las 3 peores del subconjunto de evaluación, con el
# contorno manual (verde) y el propio (rojo), mostrando la ventana del método.
# ---------------------------------------------------------------------------
titulo(f"BLOQUE 5. Segmentaciones del mejor método ({mejor})")

lado = int(elegidos.loc[mejor, "ventana"])
tolerancia = elegidos.loc[mejor, "tolerancia"]
ruta_parametros = os.path.join(CARPETA_SALIDA, "parametros_elegidos.csv")
pd.DataFrame([{"metodo": mejor, "ventana": lado, "tolerancia": tolerancia}]).to_csv(ruta_parametros, index=False)
print(f"Método y parámetros guardados en {ruta_parametros} (los usa la interfaz)")
para_figura = {}
ids_figura = list(ordenadas.index[-3:][::-1]) + list(ordenadas.index[:3])

for _, fila in df.iterrows():
    limpia, con_lee, lesion = preparar(fila)
    clic, referencia = clic_y_referencia(lesion)
    completa, ventana = segmentar_completa(mejor, con_lee, clic, lado, tolerancia)
    Image.fromarray((completa * 255).astype(np.uint8)).save(
        os.path.join(CARPETA_SEGMENTACIONES, fila["id"] + ".png"))
    if fila["id"] in ids_figura:
        para_figura[fila["id"]] = (limpia[ventana], referencia[ventana], completa[ventana])
print(f"{N} segmentaciones guardadas en {CARPETA_SEGMENTACIONES}/")

fig, ejes = plt.subplots(2, 3, figsize=(14, 9.5))
for eje, id_imagen in zip(ejes.ravel(), ids_figura):
    imagen, referencia, propia = para_figura[id_imagen]
    eje.imshow(imagen, cmap="gray", vmin=0, vmax=255)
    eje.contour(referencia, levels=[0.5], colors="lime", linewidths=1.5)
    if propia.any():
        eje.contour(propia, levels=[0.5], colors="red", linewidths=1.5)
    eje.set_title(f"{id_imagen}: Dice = {numero(dice_eval.loc[id_imagen, mejor], 3)}", fontsize=10)
    eje.axis("off")
plt.suptitle(f"{mejor}: 3 mejores (arriba) y 3 peores (abajo). Verde = manual, rojo = propia")
plt.tight_layout()
guardar_figura("05_mejores_y_peores.png", dpi=110)