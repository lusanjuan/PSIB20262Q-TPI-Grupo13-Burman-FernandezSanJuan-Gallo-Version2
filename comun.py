# comun.py
# Cosas que usan varios pasos: la tabla de imágenes y la función para cargarlas.
# Se importa así:  from comun import armar_tabla, cargar

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
