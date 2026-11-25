# paso1_exploracion.py
# Exploración inicial del dataset BUSI (ecografías de mama).
# Se corre desde la carpeta del proyecto:  python3 paso1_exploracion.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso1_exploracion/.
#
# Figuras que genera:
#   02_imagenes_por_clase.png          barras con la cantidad de imágenes por clase
#   04_ejemplos_por_clase.png          4 ejemplos por clase con el borde de la lesión
#   05a_tamanio_imagenes.png           alto contra ancho de cada imagen
#   05b_fraccion_lesion.png            fracción de la imagen ocupada por la lesión
#   06_imagenes_con_color.png          imágenes con color real (marcas, Doppler)
#   07_varias_lesiones.png             imágenes cuya máscara tiene más de una mancha
#   08_histograma_lesion_vs_anillo.png histograma de brillo de la lesión y del anillo
#   08a_ejemplo_anillo.png             cómo se arma el anillo de tejido, sobre un ejemplo
#   08b_contraste_por_clase.png        contraste lesión/anillo de cada imagen, por clase
#   08c_mediana_lesion_vs_anillo.png   mediana de la lesión contra la del anillo
#   09_benign_240_255_277.png          las tres imágenes casi iguales, lado a lado
#   09b_diferencias_240_255_277.png    diferencia píxel a píxel entre esas tres
#   10a_parecido_maximo.png            parecido de cada imagen con la más parecida
#   10b_pares_mas_parecidos.png        los pares más parecidos de todo el dataset
#   10c_pares_por_rango.png            pares en cada rango de parecido (para elegir el umbral)
#   10d_pares_clases_distintas.png     pares parecidos con clases distintas
# Tablas:
#   tabla_por_imagen.csv               una fila por imagen con todo lo medido acá
#   posibles_repetidas.csv             pares de imágenes iguales o casi iguales
#   grupos_repetidas.csv               grupo de cada imagen y si se excluye (lo usa comun.py)

import os
import hashlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.stats import mannwhitneyu

from comun import CLASES, armar_tabla
from comun import cargar as cargar_del_disco

# Cada imagen se lee del disco una sola vez y queda guardada en memoria: varios
# bloques recorren las 780 imágenes, y leerlas de nuevo cada vez es lo más lento.
_memoria = {}


def cargar(fila):
    """Igual que cargar() de comun.py, pero lee cada imagen del disco una sola vez."""
    if fila["id"] not in _memoria:
        _memoria[fila["id"]] = cargar_del_disco(fila)
    return _memoria[fila["id"]]

CARPETA_SALIDA = "outputs/paso1_exploracion"

os.makedirs(CARPETA_SALIDA, exist_ok=True)

# Un color fijo por clase, el mismo en todas las figuras
COLORES = {"benign": "#2a78b5", "malignant": "#c8553d", "normal": "#8c8c8c"}


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


def comparar(df, columna, nombre):
    """Compara una columna entre benignas y malignas con Mann-Whitney e imprime el resultado.
    Devuelve el p-valor (para ponerlo en las figuras)."""
    benignas = df.loc[df["clase"] == "benign", columna].dropna()
    malignas = df.loc[df["clase"] == "malignant", columna].dropna()
    estadistico, p = mannwhitneyu(benignas, malignas, alternative="two-sided")
    print(f"\n{nombre}")
    print("  Test: Mann-Whitney U (dos colas), benignas vs. malignas")
    print(f"  n benignas = {len(benignas)}   n malignas = {len(malignas)}")
    print(f"  Mediana benignas = {numero(benignas.median())}   Mediana malignas = {numero(malignas.median())}")
    print(f"  U = {numero(estadistico, 1)}   {formato_p(p)}")
    return p


def guardar_figura(nombre, dpi=120):
    """Guarda la figura actual en la carpeta de salida, la cierra e imprime la ruta."""
    ruta = os.path.join(CARPETA_SALIDA, nombre)
    plt.savefig(ruta, dpi=dpi)
    plt.close()
    print(f"Figura guardada en {ruta}")


def limpiar_ejes(eje):
    """Saca los bordes de arriba y de la derecha (dejan la figura más limpia)."""
    eje.spines[["top", "right"]].set_visible(False)


def cajas_con_puntos(eje, valores_por_clase, clases):
    """Boxplot por clase con cada imagen como un punto encima (con un poco de
    desplazamiento horizontal al azar para que no se tapen)."""
    rng = np.random.default_rng(0)
    eje.boxplot([valores_por_clase[c] for c in clases], widths=0.5, showfliers=False,
                medianprops={"color": "black", "linewidth": 2})
    for i, c in enumerate(clases, start=1):
        v = valores_por_clase[c]
        eje.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, alpha=0.45, color=COLORES[c],
                    edgecolors="none")
    eje.set_xticks(range(1, len(clases) + 1), [f"{c}\n(n = {len(valores_por_clase[c])})" for c in clases])
    limpiar_ejes(eje)


# ---------------------------------------------------------------------------
# BLOQUE 1. Tabla con una fila por imagen.
# Recorremos cada carpeta de clase y nos quedamos solo con las imágenes reales
# (los archivos que tienen "_mask" en el nombre son máscaras, NO imágenes).
# Para cada imagen buscamos todas sus máscaras: "_mask.png", "_mask_1.png", etc.
# Ordenamos por el número entre paréntesis para que quede 1, 2, 3... y no 1, 10, 100.
# ---------------------------------------------------------------------------
titulo("BLOQUE 1. Tabla de imágenes")

df = armar_tabla()   # la función está en comun.py

print(f"Total de imágenes: {len(df)}")
print(f"Imágenes sin ninguna máscara: {(df['rutas_mascaras'].apply(len) == 0).sum()}")
print("\nPrimeras filas:")
print(df[["id", "clase", "ruta_imagen"]].head())


# ---------------------------------------------------------------------------
# BLOQUE 2. Cantidad de imágenes por clase.
# Sirve para ver si las clases están balanceadas (importa para entrenar y evaluar).
# Figura: barras con la cantidad y el porcentaje de cada clase.
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Imágenes por clase")

conteo = df["clase"].value_counts().reindex(CLASES)
for clase in CLASES:
    porcentaje = 100 * conteo[clase] / len(df)
    print(f"  {clase:10s} {conteo[clase]:4d} imágenes  ({numero(porcentaje, 1)} %)")

fig, eje = plt.subplots(figsize=(6, 4))
barras = eje.bar(CLASES, conteo.values, color=[COLORES[c] for c in CLASES], width=0.6)
for barra, clase in zip(barras, CLASES):
    eje.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 5,
             f"{conteo[clase]} ({numero(100 * conteo[clase] / len(df), 1)} %)", ha="center", va="bottom")
eje.set_ylabel("Cantidad de imágenes")
eje.set_ylim(0, conteo.max() * 1.15)
eje.set_title(f"Imágenes por clase (total: {len(df)})")
limpiar_ejes(eje)
plt.tight_layout()
guardar_figura("02_imagenes_por_clase.png")


# ---------------------------------------------------------------------------
# BLOQUE 3. Función para cargar una imagen y su máscara.
# La imagen se lee en escala de grises (valores 0 a 255).
# Si hay varias máscaras (varias lesiones), se unen en una sola con un "o" lógico.
# La máscara queda como True (lesión) / False (fondo).
# ---------------------------------------------------------------------------
# La función cargar(fila) está en comun.py (se importa arriba).


# ---------------------------------------------------------------------------
# BLOQUE 4. Figura con 4 ejemplos por clase y el borde de la lesión en rojo.
# Sirve para ver cómo lucen las imágenes y si las máscaras coinciden con la lesión.
# Elegimos los ejemplos al azar con semilla fija para que siempre salgan los mismos.
# ---------------------------------------------------------------------------
titulo("BLOQUE 4. Ejemplos por clase")

fig, ejes = plt.subplots(3, 4, figsize=(14, 10))
for i, clase in enumerate(CLASES):
    ejemplos = df[df["clase"] == clase].sample(4, random_state=0)
    for j, (_, fila) in enumerate(ejemplos.iterrows()):
        imagen, mascara = cargar(fila)
        eje = ejes[i, j]
        eje.imshow(imagen, cmap="gray", vmin=0, vmax=255)
        if mascara.any():
            eje.contour(mascara, levels=[0.5], colors="red", linewidths=1.5)
        eje.set_title(fila["id"], fontsize=10)
        eje.axis("off")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "04_ejemplos_por_clase.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"Figura guardada en {ruta_figura}")


# ---------------------------------------------------------------------------
# BLOQUE 5. Tamaño de las imágenes y tamaño de la lesión.
# Para cada imagen guardamos alto, ancho y qué fracción de la imagen ocupa la lesión.
# Luego comparamos benignas vs. malignas con Mann-Whitney (no asume normalidad,
# y estas medidas suelen ser asimétricas).
# Las normales no tienen lesión, por eso no entran en la comparación.
# Figuras:
#   05a: alto contra ancho, un punto por imagen. Muestra que las imágenes no
#        tienen todas el mismo tamaño (hay que tenerlo en cuenta al comparar).
#   05b: fracción de la lesión por clase. Escala logarítmica porque va de
#        lesiones diminutas (0,1 %) a lesiones que ocupan media imagen.
# ---------------------------------------------------------------------------
titulo("BLOQUE 5. Tamaño de imagen y de lesión")

altos, anchos, fracciones = [], [], []
for _, fila in df.iterrows():
    imagen, mascara = cargar(fila)
    altos.append(imagen.shape[0])
    anchos.append(imagen.shape[1])
    fracciones.append(mascara.mean())
df["alto"] = altos
df["ancho"] = anchos
df["fraccion_lesion"] = fracciones

print("Resumen de alto y ancho (píxeles) por clase:")
print(df.groupby("clase")[["alto", "ancho"]].describe().round(0).T.to_string())

comparar(df, "alto", "Alto de la imagen (píxeles)")
comparar(df, "ancho", "Ancho de la imagen (píxeles)")
p_fraccion = comparar(df, "fraccion_lesion", "Fracción de la imagen ocupada por la lesión")

print(f"\nNormales con máscara no vacía: {(df.loc[df['clase'] == 'normal', 'fraccion_lesion'] > 0).sum()}")

# 05a: alto contra ancho
fig, eje = plt.subplots(figsize=(7, 6))
for clase in CLASES:
    parte = df[df["clase"] == clase]
    eje.scatter(parte["ancho"], parte["alto"], s=14, alpha=0.6, color=COLORES[clase],
                edgecolors="none", label=f"{clase} (n = {len(parte)})")
eje.set_xlabel("Ancho (píxeles)")
eje.set_ylabel("Alto (píxeles)")
eje.set_title("Tamaño de cada imagen")
eje.legend()
limpiar_ejes(eje)
plt.tight_layout()
guardar_figura("05a_tamanio_imagenes.png")

# 05b: fracción de la lesión por clase (en %)
con_lesion = ["benign", "malignant"]
valores = {c: 100 * df.loc[df["clase"] == c, "fraccion_lesion"].to_numpy() for c in con_lesion}
fig, eje = plt.subplots(figsize=(6, 5))
cajas_con_puntos(eje, valores, con_lesion)
eje.set_yscale("log")
eje.set_ylabel("Porcentaje de la imagen ocupado por la lesión (escala log)")
eje.set_title(f"Tamaño de la lesión (Mann-Whitney: {formato_p(p_fraccion)})")
plt.tight_layout()
guardar_figura("05b_fraccion_lesion.png")


# ---------------------------------------------------------------------------
# BLOQUE 6. Imágenes con color real.
# Las ecografías deberían ser grises (R = G = B en todos los píxeles).
# Si algún píxel tiene canales distintos, la imagen tiene color real
# (por ejemplo, marcas de colores o Doppler). Conviene saberlo antes de pasar a gris.
# ---------------------------------------------------------------------------
titulo("BLOQUE 6. Imágenes con color real")

tiene_color = []
for _, fila in df.iterrows():
    rgb = np.array(Image.open(fila["ruta_imagen"]).convert("RGB")).astype(int)
    distinto = (rgb[:, :, 0] != rgb[:, :, 1]) | (rgb[:, :, 1] != rgb[:, :, 2])
    tiene_color.append(distinto.any())
df["tiene_color"] = tiene_color

print(f"Imágenes con color real: {df['tiene_color'].sum()} de {len(df)}")
print(df.groupby("clase")["tiene_color"].sum().reindex(CLASES).to_string())

con_color = df[df["tiene_color"]].head(8)
if len(con_color) > 0:
    fig, ejes = plt.subplots(1, len(con_color), figsize=(3.5 * len(con_color), 4), squeeze=False)
    for j, (_, fila) in enumerate(con_color.iterrows()):
        ejes[0, j].imshow(Image.open(fila["ruta_imagen"]).convert("RGB"))
        ejes[0, j].set_title(fila["id"], fontsize=10)
        ejes[0, j].axis("off")
    plt.tight_layout()
    ruta_figura = os.path.join(CARPETA_SALIDA, "06_imagenes_con_color.png")
    plt.savefig(ruta_figura, dpi=120)
    plt.close()
    print(f"Figura guardada en {ruta_figura}")
print("Lista:", ", ".join(df.loc[df["tiene_color"], "id"]))


# ---------------------------------------------------------------------------
# BLOQUE 7. Máscaras con más de una lesión.
# Lo miramos de dos formas:
#   a) imágenes que vienen con más de un archivo de máscara;
#   b) imágenes cuya máscara unida tiene más de una mancha separada
#      (componentes conectadas), aunque venga en un solo archivo.
#      Ignoramos manchas de menos de 50 píxeles: son puntitos sueltos que
#      quedaron al dibujar la máscara, no lesiones.
# Figura: hasta 8 imágenes con más de una mancha, cada mancha con un color.
# ---------------------------------------------------------------------------
titulo("BLOQUE 7. Imágenes con más de una lesión")

MIN_PIXELES = 50
df["n_archivos_mascara"] = df["rutas_mascaras"].apply(len)
manchas = []
puntitos = 0
for _, fila in df.iterrows():
    _, mascara = cargar(fila)
    etiquetas, n = ndimage.label(mascara)
    tamanios = np.bincount(etiquetas.ravel())[1:]   # píxeles de cada mancha
    manchas.append(int((tamanios >= MIN_PIXELES).sum()))
    puntitos += int((tamanios < MIN_PIXELES).sum())
df["n_manchas"] = manchas

print("a) Con más de un archivo de máscara:")
print(df[df["n_archivos_mascara"] > 1].groupby("clase").size().reindex(CLASES, fill_value=0).to_string())
print(f"   Total: {(df['n_archivos_mascara'] > 1).sum()}")
print(f"\nb) Con más de una mancha separada en la máscara unida (manchas >= {MIN_PIXELES} píxeles):")
print(df[df["n_manchas"] > 1].groupby("clase").size().reindex(CLASES, fill_value=0).to_string())
print(f"   Total: {(df['n_manchas'] > 1).sum()}")
print(f"   (Se ignoraron {puntitos} puntitos de menos de {MIN_PIXELES} píxeles.)")

varias = df[df["n_manchas"] > 1].head(8)
if len(varias) > 0:
    colores_manchas = ["red", "yellow", "cyan", "lime", "magenta", "orange"]
    columnas = min(4, len(varias))
    filas_fig = int(np.ceil(len(varias) / columnas))
    fig, ejes = plt.subplots(filas_fig, columnas, figsize=(4 * columnas, 4 * filas_fig), squeeze=False)
    for eje in ejes.ravel():
        eje.axis("off")
    for eje, (_, fila) in zip(ejes.ravel(), varias.iterrows()):
        imagen, mascara = cargar(fila)
        etiquetas, n = ndimage.label(mascara)
        tamanios = np.bincount(etiquetas.ravel())
        eje.imshow(imagen, cmap="gray", vmin=0, vmax=255)
        k = 0
        for etiqueta in range(1, n + 1):
            if tamanios[etiqueta] >= MIN_PIXELES:
                eje.contour(etiquetas == etiqueta, levels=[0.5],
                            colors=colores_manchas[k % len(colores_manchas)], linewidths=1.5)
                k += 1
        eje.set_title(f"{fila['id']}: {fila['n_manchas']} manchas, "
                      f"{fila['n_archivos_mascara']} archivo(s)", fontsize=9)
    plt.suptitle("Máscaras con más de una lesión (un color por mancha)")
    plt.tight_layout()
    guardar_figura("07_varias_lesiones.png")


# ---------------------------------------------------------------------------
# BLOQUE 8. Brillo de la lesión vs. tejido de alrededor.
# Anillo = máscara agrandada con un disco de radio 15 píxeles, menos la máscara.
# Así comparamos la lesión con el tejido que la rodea en la misma imagen.
# Guardamos el histograma de brillo de cada región (lesión y anillo) por clase,
# y calculamos el contraste por imagen = mediana del anillo − mediana de la lesión
# (positivo = la lesión es más oscura que su entorno).
# Solo benignas y malignas: las normales no tienen lesión.
# Figuras extra:
#   08a: cómo se ve el anillo sobre una imagen de ejemplo.
#   08b: el contraste de cada imagen, por clase (la línea en 0 separa lesiones
#        más oscuras, arriba, de lesiones más claras, abajo).
#   08c: mediana de la lesión contra mediana del anillo, un punto por imagen.
#        Los puntos por encima de la diagonal son lesiones más oscuras que su entorno.
# ---------------------------------------------------------------------------
titulo("BLOQUE 8. Brillo de la lesión vs. tejido de alrededor")

# Disco de radio 15: los puntos (x, y) con x² + y² <= 15²
radio = 15
y, x = np.ogrid[-radio:radio + 1, -radio:radio + 1]
disco = x ** 2 + y ** 2 <= radio ** 2

# Histogramas acumulados (256 niveles de gris) por clase y región
histogramas = {c: {"lesion": np.zeros(256), "anillo": np.zeros(256)} for c in ["benign", "malignant"]}
contrastes = []
medianas_lesion, medianas_anillo = [], []

for _, fila in df.iterrows():
    if fila["clase"] == "normal":
        contrastes.append(np.nan)
        medianas_lesion.append(np.nan)
        medianas_anillo.append(np.nan)
        continue
    imagen, mascara = cargar(fila)
    if not mascara.any():
        contrastes.append(np.nan)
        medianas_lesion.append(np.nan)
        medianas_anillo.append(np.nan)
        continue
    anillo = ndimage.binary_dilation(mascara, structure=disco) & ~mascara
    pix_lesion = imagen[mascara]
    pix_anillo = imagen[anillo]
    histogramas[fila["clase"]]["lesion"] += np.bincount(pix_lesion, minlength=256)
    histogramas[fila["clase"]]["anillo"] += np.bincount(pix_anillo, minlength=256)
    contrastes.append(np.median(pix_anillo) - np.median(pix_lesion))
    medianas_lesion.append(np.median(pix_lesion))
    medianas_anillo.append(np.median(pix_anillo))
df["contraste"] = contrastes
df["mediana_lesion"] = medianas_lesion
df["mediana_anillo"] = medianas_anillo

fig, ejes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for eje, clase in zip(ejes, ["benign", "malignant"]):
    for region, color in [("lesion", "tab:red"), ("anillo", "tab:blue")]:
        h = histogramas[clase][region]
        eje.plot(np.arange(256), h / h.sum(), color=color, label=region)
    eje.set_title(clase)
    eje.set_xlabel("Nivel de gris (0–255)")
    eje.legend()
ejes[0].set_ylabel("Proporción de píxeles")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "08_histograma_lesion_vs_anillo.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"Figura guardada en {ruta_figura}")

print("\nMediana de brillo por región (todas las imágenes juntas):")
for clase in ["benign", "malignant"]:
    for region in ["lesion", "anillo"]:
        acumulado = np.cumsum(histogramas[clase][region])
        mediana = np.searchsorted(acumulado, acumulado[-1] / 2)
        print(f"  {clase:10s} {region:7s} {mediana}")

print(f"\nImágenes con lesión más oscura que su entorno (contraste > 0): "
      f"{(df['contraste'] > 0).sum()} de {df['contraste'].notna().sum()}")
p_contraste = comparar(df, "contraste", "Contraste = mediana del anillo − mediana de la lesión")

# 08a: el anillo sobre un ejemplo (imagen entera y recorte alrededor de la lesión)
ID_EJEMPLO_ANILLO = "benign (209)"
if ID_EJEMPLO_ANILLO not in set(df["id"]):
    ID_EJEMPLO_ANILLO = df.loc[df["clase"] == "benign", "id"].iloc[0]
fila = df[df["id"] == ID_EJEMPLO_ANILLO].iloc[0]
imagen, mascara = cargar(fila)
anillo = ndimage.binary_dilation(mascara, structure=disco) & ~mascara
filas_l, columnas_l = np.nonzero(mascara)
margen = 40
r0, r1 = max(filas_l.min() - margen, 0), min(filas_l.max() + margen, imagen.shape[0])
c0, c1 = max(columnas_l.min() - margen, 0), min(columnas_l.max() + margen, imagen.shape[1])

fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
for eje, (zona, nombre) in zip(ejes, [((slice(None), slice(None)), "imagen entera"),
                                      ((slice(r0, r1), slice(c0, c1)), "recorte")]):
    eje.imshow(imagen[zona], cmap="gray", vmin=0, vmax=255)
    eje.contour(mascara[zona], levels=[0.5], colors="red", linewidths=1.5)
    eje.contour((mascara | anillo)[zona], levels=[0.5], colors="cyan", linewidths=1.5)
    eje.set_title(f"{ID_EJEMPLO_ANILLO}, {nombre}")
    eje.axis("off")
contraste_ejemplo = df.loc[df["id"] == ID_EJEMPLO_ANILLO, "contraste"].iloc[0]
plt.suptitle(f"Rojo: borde de la lesión. Celeste: borde externo del anillo (radio {radio} px). "
             f"Contraste de esta imagen = {numero(contraste_ejemplo, 1)}")
plt.tight_layout()
guardar_figura("08a_ejemplo_anillo.png")

# 08b: contraste por clase
valores = {c: df.loc[df["clase"] == c, "contraste"].dropna().to_numpy() for c in con_lesion}
fig, eje = plt.subplots(figsize=(6, 5))
cajas_con_puntos(eje, valores, con_lesion)
eje.axhline(0, color="black", linestyle="--", linewidth=1)
eje.set_ylabel("Contraste = mediana del anillo − mediana de la lesión")
eje.set_title(f"Contraste lesión / tejido (Mann-Whitney: {formato_p(p_contraste)})")
plt.tight_layout()
guardar_figura("08b_contraste_por_clase.png")

# 08c: mediana de la lesión contra mediana del anillo
fig, eje = plt.subplots(figsize=(6, 6))
for clase in con_lesion:
    parte = df[(df["clase"] == clase) & df["contraste"].notna()]
    eje.scatter(parte["mediana_lesion"], parte["mediana_anillo"], s=12, alpha=0.6,
                color=COLORES[clase], edgecolors="none", label=f"{clase} (n = {len(parte)})")
eje.plot([0, 255], [0, 255], color="black", linestyle="--", linewidth=1, label="anillo = lesión")
eje.set_xlim(0, 255)
eje.set_ylim(0, 255)
eje.set_aspect("equal")
eje.set_xlabel("Mediana de la lesión (nivel de gris)")
eje.set_ylabel("Mediana del anillo (nivel de gris)")
eje.set_title("Arriba de la diagonal: lesión más oscura que su entorno")
eje.legend(loc="lower right")
limpiar_ejes(eje)
plt.tight_layout()
guardar_figura("08c_mediana_lesion_vs_anillo.png")


# ---------------------------------------------------------------------------
# BLOQUE 9. Posibles imágenes repetidas.
# benign (240), (255) y (277) parecen casi iguales. Las ponemos lado a lado
# para verlo; si son duplicados, no deberían quedar repartidas entre
# entrenamiento y prueba (eso inflaría los resultados).
# Figura extra (09b): la diferencia absoluta píxel a píxel entre cada par.
# Si dos imágenes son la misma, la diferencia es negra salvo en lo que cambia
# (marcas, calipers). Si tienen distinto tamaño, la segunda se redimensiona al
# tamaño de la primera para poder restarlas (y se avisa en el título).
# ---------------------------------------------------------------------------
titulo("BLOQUE 9. Imágenes casi iguales")

ids = ["benign (240)", "benign (255)", "benign (277)"]
fig, ejes = plt.subplots(1, 3, figsize=(14, 5))
for eje, id_imagen in zip(ejes, ids):
    fila = df[df["id"] == id_imagen].iloc[0]
    imagen, mascara = cargar(fila)
    eje.imshow(imagen, cmap="gray", vmin=0, vmax=255)
    if mascara.any():
        eje.contour(mascara, levels=[0.5], colors="red", linewidths=1.5)
    eje.set_title(f"{id_imagen}  ({imagen.shape[0]}×{imagen.shape[1]})", fontsize=10)
    eje.axis("off")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "09_benign_240_255_277.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"Figura guardada en {ruta_figura}")

pares = [(ids[0], ids[1]), (ids[0], ids[2]), (ids[1], ids[2])]
fig, ejes = plt.subplots(1, 3, figsize=(14, 5))
for eje, (id_a, id_b) in zip(ejes, pares):
    a, _ = cargar(df[df["id"] == id_a].iloc[0])
    b, _ = cargar(df[df["id"] == id_b].iloc[0])
    aviso = ""
    if a.shape != b.shape:
        b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.BILINEAR))
        aviso = " (redimensionada)"
    diferencia = np.abs(a.astype(int) - b.astype(int))
    print(f"  {id_a} vs. {id_b}{aviso}: diferencia media = {numero(diferencia.mean(), 1)} niveles de gris, "
          f"píxeles iguales = {numero(100 * (diferencia == 0).mean(), 1)} %")
    eje.imshow(diferencia, cmap="gray", vmin=0, vmax=100)
    eje.set_title(f"|{id_a} − {id_b}|{aviso}\nmedia = {numero(diferencia.mean(), 1)}", fontsize=10)
    eje.axis("off")
plt.suptitle("Diferencia absoluta entre las imágenes casi iguales (negro = iguales; blanco = 100 o más)")
plt.tight_layout()
guardar_figura("09b_diferencias_240_255_277.png")


# ---------------------------------------------------------------------------
# BLOQUE 10. Búsqueda de imágenes repetidas en todo el dataset.
# El bloque 9 mira tres imágenes que encontramos a ojo. Acá buscamos en las 780:
#   a) Repetidas exactas: mismos píxeles. Para cada imagen se calcula un "hash"
#      (una huella de 32 caracteres que cambia si cambia un solo píxel) y se
#      buscan huellas iguales. Si las dos copias tienen distinta clase, la
#      etiqueta es contradictoria.
#   b) Casi repetidas: la misma escena con pequeños cambios (marcas, recorte).
#      Cada imagen se achica a 32 × 32 y se normaliza (media 0, desvío 1); el
#      parecido entre dos imágenes es la correlación entre sus miniaturas
#      (1 = iguales). El umbral (0,95) se eligió mirando la figura 10c: aun los
#      pares menos parecidos por encima de 0,95 son la misma lesión.
#   c) Grupos: si A se parece a B y B a C, las tres son la misma ecografía.
#      Los grupos con clases distintas (etiqueta contradictoria) se excluyen de
#      los pasos siguientes; las demás repetidas se quedan, y el número de grupo
#      se usa en la validación cruzada.
# La búsqueda se hace una vez y se guarda en grupos_repetidas.csv; las corridas
# siguientes la leen de ahí (RECALCULAR_REPETIDAS = True para rehacerla).
# Para el clasificador: las repetidas tienen que caer en el mismo grupo de la
# validación cruzada (o sacarse), si no, el modelo "ve" en el test imágenes que
# ya vio en el entrenamiento y el resultado sale inflado.
# ---------------------------------------------------------------------------
titulo("BLOQUE 10. Imágenes repetidas en todo el dataset")

UMBRAL_PARECIDO = 0.95   # elegido mirando la figura 10c: hasta 0,95 todos los pares son la misma lesión
RECALCULAR_REPETIDAS = False   # True: vuelve a buscar las repetidas (tarda y rehace las figuras 10a-10d)
RUTA_GRUPOS = os.path.join(CARPETA_SALIDA, "grupos_repetidas.csv")

if os.path.exists(RUTA_GRUPOS) and not RECALCULAR_REPETIDAS:
    grupos = pd.read_csv(RUTA_GRUPOS)
    df = df.merge(grupos[["id", "grupo_repetidas", "excluir"]], on="id", how="left")
    print(f"Grupos leídos de {RUTA_GRUPOS} (no se recalculan; las figuras 10a a 10d son de la corrida anterior)")
else:
    LADO_MINIATURA = 32

    huellas, miniaturas = [], []
    for _, fila in df.iterrows():
        imagen, _ = cargar(fila)
        huellas.append(hashlib.md5(imagen.tobytes() + str(imagen.shape).encode()).hexdigest())
        mini = np.array(Image.fromarray(imagen).resize((LADO_MINIATURA, LADO_MINIATURA), Image.BILINEAR),
                        dtype=float).ravel()
        miniaturas.append((mini - mini.mean()) / (mini.std() + 1e-9))
    df["hash"] = huellas

    # a) exactas
    print("a) Repetidas exactas (mismos píxeles):")
    grupos = df.groupby("hash")["id"].apply(list)
    grupos = grupos[grupos.apply(len) > 1]
    if len(grupos) == 0:
        print("   Ninguna.")
    for lista in grupos:
        clases = df.set_index("id").loc[lista, "clase"]
        aviso = "  <- ¡CLASES DISTINTAS!" if clases.nunique() > 1 else ""
        print(f"   {', '.join(lista)}{aviso}")

    # b) casi repetidas
    M = np.array(miniaturas)
    parecido = M @ M.T / M.shape[1]          # correlación entre cada par de miniaturas
    np.fill_diagonal(parecido, -1)            # que una imagen no cuente consigo misma
    i_sup, j_sup = np.triu_indices(len(df), k=1)
    valores_pares = parecido[i_sup, j_sup]
    seleccion = valores_pares >= UMBRAL_PARECIDO
    filas_pares = []
    for i, j, r in zip(i_sup[seleccion], j_sup[seleccion], valores_pares[seleccion]):
        filas_pares.append({
            "id_a": df["id"].iloc[i], "clase_a": df["clase"].iloc[i],
            "id_b": df["id"].iloc[j], "clase_b": df["clase"].iloc[j],
            "correlacion": r, "exacta": df["hash"].iloc[i] == df["hash"].iloc[j],
        })
    tabla_pares = pd.DataFrame(filas_pares, columns=["id_a", "clase_a", "id_b", "clase_b", "correlacion", "exacta"])
    tabla_pares = tabla_pares.sort_values("correlacion", ascending=False).reset_index(drop=True)
    tabla_pares.to_csv(os.path.join(CARPETA_SALIDA, "posibles_repetidas.csv"), index=False)

    print(f"\nb) Pares con parecido >= {numero(UMBRAL_PARECIDO, 2)} (miniaturas de {LADO_MINIATURA} × {LADO_MINIATURA}): "
          f"{len(tabla_pares)}")
    print(f"   De esos, con clases distintas: {(tabla_pares['clase_a'] != tabla_pares['clase_b']).sum()}")
    for _, par in tabla_pares.head(30).iterrows():
        marca = " (exacta)" if par["exacta"] else ""
        print(f"   {par['id_a']:16s} {par['id_b']:16s} r = {numero(par['correlacion'], 3)}{marca}")
    if len(tabla_pares) > 30:
        print(f"   ... (lista completa en posibles_repetidas.csv)")

    # 10a: para cada imagen, el parecido con la imagen más parecida del dataset
    parecido_maximo = parecido.max(axis=1)
    fig, eje = plt.subplots(figsize=(8, 4))
    eje.hist(parecido_maximo, bins=60, color="#5a5a5a")
    eje.axvline(UMBRAL_PARECIDO, color="red", linestyle="--", label=f"umbral = {numero(UMBRAL_PARECIDO, 2)}")
    eje.set_yscale("log")
    eje.set_xlabel("Correlación con la imagen más parecida del dataset")
    eje.set_ylabel("Cantidad de imágenes (escala log)")
    eje.set_title("Parecido máximo de cada imagen (las repetidas quedan a la derecha, cerca de 1)")
    eje.legend()
    limpiar_ejes(eje)
    plt.tight_layout()
    guardar_figura("10a_parecido_maximo.png")

    def figura_pares(pares, nombre_archivo, titulo_figura):
        """Muestra cada par de imágenes lado a lado, con su parecido en el título."""
        if len(pares) == 0:
            print(f"(No hay pares para {nombre_archivo}.)")
            return
        fig, ejes = plt.subplots(len(pares), 2, figsize=(8, 3.2 * len(pares)), squeeze=False)
        for k, (_, par) in enumerate(pares.iterrows()):
            texto_r = f"r = {numero(par['correlacion'], 4)}" + (", exacta" if par["exacta"] else "")
            for columna, id_imagen in enumerate([par["id_a"], par["id_b"]]):
                imagen, _ = cargar(df[df["id"] == id_imagen].iloc[0])
                ejes[k, columna].imshow(imagen, cmap="gray", vmin=0, vmax=255)
                ejes[k, columna].set_title(f"{id_imagen}   ({texto_r})", fontsize=10)
                ejes[k, columna].axis("off")
        plt.suptitle(titulo_figura)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        guardar_figura(nombre_archivo, dpi=100)


    # 10b: los pares más parecidos
    figura_pares(tabla_pares.head(8), "10b_pares_mas_parecidos.png", "Pares más parecidos del dataset")

    # 10c: pares en distintos rangos de parecido, para elegir a ojo el umbral.
    # De cada rango se muestran los 2 pares MENOS parecidos (los más dudosos):
    # si esos ya son la misma ecografía, todo el rango lo es.
    RANGOS = [r for r in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995] if r >= UMBRAL_PARECIDO] + [1.0001]
    print("\nPares por rango de parecido:")
    muestras = []
    for desde, hasta in zip(RANGOS[:-1], RANGOS[1:]):
        en_rango = tabla_pares[(tabla_pares["correlacion"] >= desde) & (tabla_pares["correlacion"] < hasta)]
        print(f"   {numero(desde, 3):>6s} a {numero(min(hasta, 1), 3):>6s}: {len(en_rango):4d} pares")
        muestras.append(en_rango.sort_values("correlacion").head(2))
    figura_pares(pd.concat(muestras), "10c_pares_por_rango.png",
                 "Los 2 pares menos parecidos de cada rango (de arriba hacia abajo: más dudosos a menos dudosos)")

    # 10d: todos los pares con clases distintas (los más importantes: etiqueta contradictoria)
    distintas = tabla_pares[tabla_pares["clase_a"] != tabla_pares["clase_b"]]
    print(f"\nPares con clases distintas: {len(distintas)}")
    for _, par in distintas.iterrows():
        print(f"   {par['id_a']:16s} {par['id_b']:16s} r = {numero(par['correlacion'], 4)}")
    figura_pares(distintas, "10d_pares_clases_distintas.png", "Pares parecidos con clases distintas")

    # Grupos: si A se parece a B y B a C, las tres son la misma ecografía.
    # Se arman con las componentes conectadas del grafo de pares (cada imagen es
    # un nodo y cada par por encima del umbral, una arista).
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    posicion = {id_imagen: k for k, id_imagen in enumerate(df["id"])}
    filas_g = [posicion[a] for a in tabla_pares["id_a"]]
    columnas_g = [posicion[b] for b in tabla_pares["id_b"]]
    grafo = coo_matrix((np.ones(len(filas_g)), (filas_g, columnas_g)), shape=(len(df), len(df)))
    _, grupo = connected_components(grafo, directed=False)
    df["grupo_repetidas"] = grupo
    tamanio_grupo = df.groupby("grupo_repetidas")["id"].transform("size")
    n_grupos = df.loc[tamanio_grupo > 1, "grupo_repetidas"].nunique()
    print(f"\nGrupos de imágenes repetidas (umbral {numero(UMBRAL_PARECIDO, 2)}): {n_grupos}, "
          f"con {(tamanio_grupo > 1).sum()} imágenes en total")
    print(f"   Si de cada grupo se deja una sola imagen, se sacan {(tamanio_grupo > 1).sum() - n_grupos}.")
    print(f"   Tamaño del grupo más grande: {tamanio_grupo.max()} imágenes")

    # Grupos con clases distintas: la misma ecografía tiene dos etiquetas y no
    # sabemos cuál es la correcta. Esas imágenes se excluyen de los pasos siguientes.
    # Las demás repetidas NO se sacan: se usa el número de grupo para que cada grupo
    # quede entero dentro de un mismo fold de la validación cruzada.
    clases_por_grupo = df.groupby("grupo_repetidas")["clase"].transform("nunique")
    df["excluir"] = clases_por_grupo > 1
    df[["id", "clase", "grupo_repetidas", "excluir"]].to_csv(RUTA_GRUPOS, index=False)
    print(f"Grupos guardados en {RUTA_GRUPOS} (lo lee comun.py)")

excluidas = df[df["excluir"]].sort_values(["grupo_repetidas", "clase", "id"])
print(f"\nImágenes a excluir (grupos con clases distintas): {len(excluidas)}")
print(excluidas.groupby("clase").size().reindex(CLASES, fill_value=0).to_string())
for numero_grupo, parte in excluidas.groupby("grupo_repetidas"):
    print(f"   grupo {numero_grupo}: {', '.join(parte['id'])}")

quedan = df[~df["excluir"]]
por_clase = ", ".join(f"{c} {(quedan['clase'] == c).sum()}" for c in CLASES)
print(f"Quedan para los pasos siguientes: {len(quedan)} imágenes ({por_clase}), "
      f"en {quedan['grupo_repetidas'].nunique()} grupos distintos")


# ---------------------------------------------------------------------------
# Tabla final: una fila por imagen con todo lo medido en este paso.
# ---------------------------------------------------------------------------
columnas = ["id", "clase", "alto", "ancho", "fraccion_lesion", "tiene_color", "n_archivos_mascara",
            "n_manchas", "mediana_lesion", "mediana_anillo", "contraste", "grupo_repetidas", "excluir"]
ruta_tabla = os.path.join(CARPETA_SALIDA, "tabla_por_imagen.csv")
df[columnas].to_csv(ruta_tabla, index=False)
print(f"\nTabla guardada en {ruta_tabla}")

print("\nListo.")