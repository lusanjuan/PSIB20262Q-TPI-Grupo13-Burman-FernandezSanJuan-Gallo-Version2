# perfil_ejemplo.py
# Rehace solo la figura del perfil de grises cruzando la lesión, con la imagen que quieran.
# Uso: python3 perfil_ejemplo.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun import armar_tabla, cargar
from marcas import limpiar
from filtros import filtro_media, filtro_mediana, filtro_gaussiano, filtro_lee, leer_varianza_ruido

ID_EJEMPLO = "benign (3)"   # por ejemplo "malignant (12)". Con None, lista candidatas y no grafica.
FILA = None         # None = fila del centro de la lesión; o un número de fila

df = armar_tabla()
df = df[df["clase"] != "normal"].reset_index(drop=True)

if ID_EJEMPLO is None:
    # Candidatas: lesiones y tejido más claros (ahí el speckle se ve más, paso 3 bloque 2)
    filas = []
    for _, fila in df.iterrows():
        img, lesion = cargar(fila)
        filas.append((fila["id"], np.median(img[lesion]), np.median(img[~lesion & (img > 0)])))
    filas.sort(key=lambda t: t[1], reverse=True)
    print("Las 15 imágenes con la lesión más clara (id, mediana lesión, mediana resto):")
    for id_img, med_les, med_resto in filas[:15]:
        print(f"  {id_img:18s} {med_les:6.0f} {med_resto:6.0f}")
    raise SystemExit

img, lesion = cargar(df[df["id"] == ID_EJEMPLO].iloc[0])
limpia, _ = limpiar(img)
versiones = {
    "sin filtrar": limpia.astype(float),
    "media": filtro_media(limpia),
    "mediana": filtro_mediana(limpia),
    "gaussiano": filtro_gaussiano(limpia),
    "lee": filtro_lee(limpia, leer_varianza_ruido()),
}

filas_lesion, columnas_lesion = np.where(lesion)
fila_perfil = int(np.round(filas_lesion.mean())) if FILA is None else FILA
x0 = max(columnas_lesion.min() - 60, 0)
x1 = min(columnas_lesion.max() + 60, img.shape[1])
columnas = np.arange(x0, x1)
bordes = np.where(np.diff(lesion[fila_perfil, columnas].astype(int)) != 0)[0] + x0 + 0.5

fig, eje = plt.subplots(figsize=(11, 4.5))
estilos = {"sin filtrar": ("lightgray", 1), "media": ("tab:orange", 1.8), "mediana": ("tab:green", 1.8),
           "gaussiano": ("tab:blue", 1.8), "lee": ("tab:red", 1.8)}
for nombre, (color, ancho) in estilos.items():
    eje.plot(columnas, versiones[nombre][fila_perfil, columnas], color=color, lw=ancho, label=nombre)
for b in bordes:
    eje.axvline(b, color="black", linestyle="--", lw=1)
eje.set_xlabel("Columna (píxeles)")
eje.set_ylabel("Nivel de gris")
eje.set_title(f"{ID_EJEMPLO}: perfil por la fila {fila_perfil} "
              "(líneas punteadas = borde de la lesión según la máscara)")
eje.legend(fontsize=8)
plt.tight_layout()
os.makedirs("outputs/paso4", exist_ok=True)
ruta = f"outputs/paso4/01b_perfil_borde_{ID_EJEMPLO.replace(' ', '_')}.png"
plt.savefig(ruta, dpi=120)
plt.close()
print(f"Figura guardada en {ruta}")