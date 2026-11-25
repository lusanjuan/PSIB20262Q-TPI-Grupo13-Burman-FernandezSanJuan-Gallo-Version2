# comun.py
# Cosas que usan varios pasos: la tabla de imágenes, la función para cargarlas,
# los grupos de imágenes repetidas (paso 1) y las imágenes limpias (paso 2).
# Se importa así:  from comun import armar_tabla, cargar, agregar_grupos, cargar_limpia

import os
import glob
import re

import numpy as np
import pandas as pd
from PIL import Image

CARPETA_DATOS = "data/raw/Dataset_BUSI_with_GT"
CLASES = ["benign", "malignant", "normal"]


# ---------------------------------------------------------------------------
# Tabla con una fila por imagen.
# Recorremos cada carpeta de clase y nos quedamos solo con las imágenes reales
# (los archivos que tienen "_mask" en el nombre son máscaras, NO imágenes).
# Para cada imagen buscamos todas sus máscaras: "_mask.png", "_mask_1.png", etc.
# Ordenamos por el número entre paréntesis para que quede 1, 2, 3... y no 1, 10, 100.
# ---------------------------------------------------------------------------
def armar_tabla():
    filas = []
    for clase in CLASES:
        archivos = glob.glob(os.path.join(CARPETA_DATOS, clase, "*.png"))
        imagenes = [a for a in archivos if "_mask" not in a]
        for ruta in imagenes:
            id_imagen = os.path.basename(ruta).replace(".png", "")      # ej: "benign (1)"
            numero_imagen = int(re.search(r"\((\d+)\)", id_imagen).group(1))
            # Máscaras de esta imagen: "benign (1)_mask.png" y "benign (1)_mask_N.png"
            mascaras = sorted(glob.glob(os.path.join(CARPETA_DATOS, clase, glob.escape(id_imagen) + "_mask*.png")))
            filas.append({
                "id": id_imagen,
                "clase": clase,
                "numero": numero_imagen,
                "ruta_imagen": ruta,
                "rutas_mascaras": mascaras,
            })

    df = pd.DataFrame(filas)
    df = df.sort_values(["clase", "numero"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Cargar una imagen y su máscara.
# La imagen se lee en escala de grises (valores 0 a 255).
# Si hay varias máscaras (varias lesiones), se unen en una sola con un "o" lógico.
# La máscara queda como True (lesión) / False (fondo).
# ---------------------------------------------------------------------------
def cargar(fila):
    imagen = np.array(Image.open(fila["ruta_imagen"]).convert("L"))
    mascara = np.zeros(imagen.shape, dtype=bool)
    for ruta_mascara in fila["rutas_mascaras"]:
        m = np.array(Image.open(ruta_mascara).convert("L")) > 0
        mascara = mascara | m
    return imagen, mascara


# ---------------------------------------------------------------------------
# Imágenes repetidas (las encuentra el paso 1, bloque 10).
# ---------------------------------------------------------------------------
RUTA_GRUPOS = "outputs/paso1_exploracion/grupos_repetidas.csv"


def agregar_grupos(df):
    """Agrega la columna 'grupo' (las imágenes repetidas comparten el número) y
    saca las imágenes cuyo grupo tiene clases distintas (etiqueta contradictoria).
    Las demás repetidas se quedan: el grupo sirve para que la validación cruzada
    no las separe entre entrenamiento y prueba. Necesita haber corrido el paso 1."""
    if not os.path.exists(RUTA_GRUPOS):
        raise FileNotFoundError(f"No existe {RUTA_GRUPOS}: corré primero paso1_exploracion.py")
    grupos = pd.read_csv(RUTA_GRUPOS)[["id", "grupo_repetidas", "excluir"]]
    df = df.merge(grupos, on="id", how="left", validate="one_to_one")
    if df["grupo_repetidas"].isna().any():
        raise ValueError("Hay imágenes sin grupo: volvé a correr paso1_exploracion.py")
    df = df[~df["excluir"].astype(bool)].drop(columns="excluir")
    return df.rename(columns={"grupo_repetidas": "grupo"}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Imágenes limpias (sin marcas del ecógrafo). Las calcula y guarda el paso 2
# una sola vez; los pasos siguientes las leen con cargar_limpia(fila), que
# devuelve lo mismo que cargar(fila) pero con la imagen ya limpia.
# Se guardan en PNG, que no pierde información.
# ---------------------------------------------------------------------------
CARPETA_LIMPIAS = "outputs/paso2_marcas/imagenes_limpias"


def ruta_limpia(fila):
    return os.path.join(CARPETA_LIMPIAS, fila["clase"], fila["id"] + ".png")


def guardar_limpia(fila, imagen):
    ruta = ruta_limpia(fila)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    Image.fromarray(imagen).save(ruta)


def cargar_limpia(fila, con_marcas=False):
    """Devuelve (imagen_limpia, mascara_lesion), como cargar(fila).
    Con con_marcas=True devuelve además la máscara de las marcas que se
    rellenaron: son justo los píxeles donde la imagen limpia difiere de la
    original (el relleno no toca nada fuera de las marcas)."""
    ruta = ruta_limpia(fila)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe {ruta}: corré primero paso2_marcas.py")
    original, mascara = cargar(fila)
    imagen = np.array(Image.open(ruta).convert("L"))
    if con_marcas:
        return imagen, mascara, imagen != original
    return imagen, mascara