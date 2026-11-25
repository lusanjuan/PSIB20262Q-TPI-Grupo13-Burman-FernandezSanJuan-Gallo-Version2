# paso1_exploracion.py
# Exploración inicial del dataset BUSI (ecografías de mama).
# Se corre desde la carpeta del proyecto:  python3 paso1_exploracion.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso1_exploracion/.

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.stats import mannwhitneyu

from comun import CLASES, armar_tabla, cargar

CARPETA_SALIDA = "outputs/paso1_exploracion"

os.makedirs(CARPETA_SALIDA, exist_ok=True)


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
    """Compara una columna entre benignas y malignas con Mann-Whitney e imprime el resultado."""
    benignas = df.loc[df["clase"] == "benign", columna].dropna()
    malignas = df.loc[df["clase"] == "malignant", columna].dropna()
    estadistico, p = mannwhitneyu(benignas, malignas, alternative="two-sided")
    print(f"\n{nombre}")
    print("  Test: Mann-Whitney U (dos colas), benignas vs. malignas")
    print(f"  n benignas = {len(benignas)}   n malignas = {len(malignas)}")
    print(f"  Mediana benignas = {numero(benignas.median())}   Mediana malignas = {numero(malignas.median())}")
    print(f"  U = {numero(estadistico, 1)}   {formato_p(p)}")


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
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Imágenes por clase")

conteo = df["clase"].value_counts().reindex(CLASES)
for clase in CLASES:
    porcentaje = 100 * conteo[clase] / len(df)
    print(f"  {clase:10s} {conteo[clase]:4d} imágenes  ({numero(porcentaje, 1)} %)")


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
comparar(df, "fraccion_lesion", "Fracción de la imagen ocupada por la lesión")

print(f"\nNormales con máscara no vacía: {(df.loc[df['clase'] == 'normal', 'fraccion_lesion'] > 0).sum()}")


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


# ---------------------------------------------------------------------------
# BLOQUE 8. Brillo de la lesión vs. tejido de alrededor.
# Anillo = máscara agrandada con un disco de radio 15 píxeles, menos la máscara.
# Así comparamos la lesión con el tejido que la rodea en la misma imagen.
# Guardamos el histograma de brillo de cada región (lesión y anillo) por clase,
# y calculamos el contraste por imagen = mediana del anillo − mediana de la lesión
# (positivo = la lesión es más oscura que su entorno).
# Solo benignas y malignas: las normales no tienen lesión.
# ---------------------------------------------------------------------------
titulo("BLOQUE 8. Brillo de la lesión vs. tejido de alrededor")

# Disco de radio 15: los puntos (x, y) con x² + y² <= 15²
radio = 15
y, x = np.ogrid[-radio:radio + 1, -radio:radio + 1]
disco = x ** 2 + y ** 2 <= radio ** 2

# Histogramas acumulados (256 niveles de gris) por clase y región
histogramas = {c: {"lesion": np.zeros(256), "anillo": np.zeros(256)} for c in ["benign", "malignant"]}
contrastes = []

for _, fila in df.iterrows():
    if fila["clase"] == "normal":
        contrastes.append(np.nan)
        continue
    imagen, mascara = cargar(fila)
    if not mascara.any():
        contrastes.append(np.nan)
        continue
    anillo = ndimage.binary_dilation(mascara, structure=disco) & ~mascara
    pix_lesion = imagen[mascara]
    pix_anillo = imagen[anillo]
    histogramas[fila["clase"]]["lesion"] += np.bincount(pix_lesion, minlength=256)
    histogramas[fila["clase"]]["anillo"] += np.bincount(pix_anillo, minlength=256)
    contrastes.append(np.median(pix_anillo) - np.median(pix_lesion))
df["contraste"] = contrastes

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
comparar(df, "contraste", "Contraste = mediana del anillo − mediana de la lesión")


# ---------------------------------------------------------------------------
# BLOQUE 9. Posibles imágenes repetidas.
# benign (240), (255) y (277) parecen casi iguales. Las ponemos lado a lado
# para verlo; si son duplicados, no deberían quedar repartidas entre
# entrenamiento y prueba (eso inflaría los resultados).
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

print("\nListo.")
