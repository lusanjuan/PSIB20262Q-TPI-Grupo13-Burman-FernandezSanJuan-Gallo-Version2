# paso2_marcas.py
# Detectar y borrar las marcas del ecógrafo (cruces de medición, líneas
# punteadas, números y textos) que vimos en el paso 1.
# Se corre desde la carpeta del proyecto:  python3 paso2_marcas.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso2/.

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import disk, opening, dilation
from skimage.restoration import inpaint_biharmonic
from scipy.stats import wilcoxon

from comun import CLASES, armar_tabla, cargar
import marcas
from marcas import detectar_marcas, limpiar

CARPETA_SALIDA = "outputs/paso2"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

df = armar_tabla()


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


def tophat_blanco(img):
    """Top-hat blanco = imagen − apertura con un disco de radio 3 (ver marcas.py)."""
    return img.astype(int) - opening(img, disk(marcas.RADIO_DISCO)).astype(int)


def rellenar_telea(img, mascara):
    """Telea: rellena desde el borde del hueco hacia adentro."""
    return cv2.inpaint(img, mascara.astype(np.uint8), 5, cv2.INPAINT_TELEA)


def rellenar_biharmonico(img, mascara):
    """Biharmónico: la superficie más suave que empalma con el borde del hueco."""
    relleno = inpaint_biharmonic(img / 255.0, mascara)
    return np.clip(np.round(relleno * 255), 0, 255).astype(np.uint8)


def fila_por_id(id_imagen):
    return df[df["id"] == id_imagen].iloc[0]


# ---------------------------------------------------------------------------
# BLOQUE 1. Por qué no sirve buscar "lo más blanco".
# La idea ingenua sería: las marcas son blancas, entonces busco píxeles = 255.
# Pero las marcas NO valen 255 (valen ~210–233), y además hay tejido que es
# todavía más brillante. Lo mostramos con:
#   - un perfil de grises (fila 227 de "malignant (12)") que cruza el brazo
#     vertical de una cruz de medición;
#   - cuántas imágenes tienen tejido (píxeles fuera de las marcas detectadas)
#     con valores >= 250. Un umbral de brillo marcaría ese tejido como marca.
# ---------------------------------------------------------------------------
titulo("BLOQUE 1. Por qué no sirve buscar lo más blanco")

img12, _ = cargar(fila_por_id("malignant (12)"))
FILA, X_INICIO, X_FIN = 227, 280, 297
print(f'Perfil de grises de "malignant (12)", fila {FILA}, columnas {X_INICIO} a {X_FIN - 1}:')
print("  columna: " + " ".join(f"{x:4d}" for x in range(X_INICIO, X_FIN)))
print("  gris:    " + " ".join(f"{v:4d}" for v in img12[FILA, X_INICIO:X_FIN]))
print("  -> La marca vale ~215–231 (no 255) y el tejido de al lado ~70–130.")
print("     Entre medio hay un borde suave (~155) que también es parte de la marca.")

imagenes_con_250 = 0
for _, fila in df.iterrows():
    img, _ = cargar(fila)
    tejido = ~detectar_marcas(img)
    if ((img >= 250) & tejido).any():
        imagenes_con_250 += 1
print(f"\nImágenes con tejido >= 250 (fuera de las marcas): {imagenes_con_250} de {len(df)}")
print("  -> Un umbral de brillo marcaría tejido como marca y no vería las marcas de ~220.")


# ---------------------------------------------------------------------------
# BLOQUE 2. Detección con top-hat blanco (función detectar_marcas en marcas.py).
# El top-hat mide cuánto sobresale cada píxel de su entorno inmediato, en vez
# de mirar el brillo absoluto. Una marca de ~220 sobre tejido de ~90 sobresale
# ~130; el tejido, aunque sea brillante, casi nunca sobresale tanto en algo
# tan angosto (3 px). La explicación con números en 1D está en marcas.py.
# Para justificar el umbral 100 imprimimos:
#   - el top-hat sobre la cruz del perfil anterior;
#   - el percentil 99,9 del top-hat en el tejido sin marcas de las 780 imágenes
#     (sin contar el fondo negro, valor 0, que tiene top-hat 0).
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Detección con top-hat blanco")

th12 = tophat_blanco(img12)
print(f'Top-hat en la misma fila de "malignant (12)":')
print("  columna: " + " ".join(f"{x:4d}" for x in range(X_INICIO, X_FIN)))
print("  top-hat: " + " ".join(f"{v:4d}" for v in th12[FILA, X_INICIO:X_FIN]))

# Histograma acumulado del top-hat (0 a 255) en el tejido de todas las imágenes
hist_tejido = np.zeros(256)
for _, fila in df.iterrows():
    img, _ = cargar(fila)
    th = tophat_blanco(img)
    tejido = ~dilation(detectar_marcas(img), disk(2)) & (img > 0)
    hist_tejido += np.bincount(th[tejido], minlength=256)
acumulado = np.cumsum(hist_tejido) / hist_tejido.sum()
p999 = np.searchsorted(acumulado, 0.999)
print(f"\nPercentil 99,9 del top-hat en el tejido sin marcas (780 imágenes): {p999}")
print(f"Píxeles de tejido con top-hat >= {marcas.UMBRAL_TOPHAT}: "
      f"{numero(100 * (1 - acumulado[marcas.UMBRAL_TOPHAT - 1]), 3)} %")
print(f"  -> Umbral = {marcas.UMBRAL_TOPHAT}: por encima de casi todo el tejido y por debajo de las marcas.")
print("     Lo poco de tejido que pasa el umbral son fibras finas, que se descartan por forma.")


# ---------------------------------------------------------------------------
# BLOQUE 3. Rellenar las marcas con 2 métodos de inpainting.
# Borrar la marca deja un hueco: hay que inventar qué tejido había debajo.
#   - Telea: rellena el hueco desde el borde hacia adentro; cada píxel nuevo es
#     un promedio de los vecinos ya conocidos, con más peso a los más cercanos.
#   - Biharmónico: rellena con la superficie más suave posible que empalma con
#     el borde del hueco.
# Figura: 3 imágenes (filas) x 4 columnas: original, marcas detectadas en rojo,
# relleno con Telea y relleno con biharmónico.
# ---------------------------------------------------------------------------
titulo("BLOQUE 3. Relleno de las marcas (Telea y biharmónico)")

ejemplos = ["malignant (12)", "benign (260)", "benign (209)"]
fig, ejes = plt.subplots(3, 4, figsize=(18, 12))
for i, id_imagen in enumerate(ejemplos):
    img, _ = cargar(fila_por_id(id_imagen))
    mascara = detectar_marcas(img)
    paneles = [
        (img, "original"),
        (img, "marcas detectadas"),
        (rellenar_telea(img, mascara), "Telea"),
        (rellenar_biharmonico(img, mascara), "biharmónico"),
    ]
    for j, (imagen, nombre) in enumerate(paneles):
        ejes[i, j].imshow(imagen, cmap="gray", vmin=0, vmax=255)
        if j == 1:
            rojo = np.zeros(img.shape + (4,))
            rojo[mascara] = [1, 0, 0, 1]
            ejes[i, j].imshow(rojo)
        ejes[i, j].set_title(f"{id_imagen} – {nombre}", fontsize=10)
        ejes[i, j].axis("off")
    print(f"  {id_imagen}: las marcas ocupan {numero(100 * mascara.mean(), 2)} % de la imagen")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "03_relleno_ejemplos.png")
plt.savefig(ruta_figura, dpi=100)
plt.close()
print(f"Figura guardada en {ruta_figura}")


# ---------------------------------------------------------------------------
# BLOQUE 4. Validación con cruces sintéticas.
# En las marcas reales no sabemos qué tejido había debajo, así que no podemos
# medir el error. Solución: dibujamos cruces nosotros sobre tejido real, lejos
# de la lesión y de las marcas reales, guardando la imagen original.
# Así sabemos la respuesta correcta.
#   - Cruz: 15 x 15 px, brazos de 3 px de grosor, valor 220 (como las reales).
#   - Trabajamos sobre un recorte de 81 x 81 px centrado en la cruz (más rápido;
#     la apertura y el relleno solo miran el entorno cercano).
# a) Detección: ¿la cruz quedó entera dentro de la máscara detectada?
#    Lo agrupamos según el brillo medio del tejido de abajo (rangos de 20 grises).
# b) Relleno: con la máscara VERDADERA de la cruz (para medir solo el relleno),
#    PSNR = 10·log10(255² / MSE), con el MSE calculado solo dentro de la cruz.
#    Más alto = el relleno se parece más a lo que había.
#    Comparamos Telea vs. biharmónico con Wilcoxon pareado (mismas cruces).
# ---------------------------------------------------------------------------
titulo("BLOQUE 4. Validación con 600 cruces sintéticas")

N_CRUCES = 600
MEDIO = 7          # la cruz va de -7 a +7 (15 px)
GROSOR = 1         # el brazo va de -1 a +1 (3 px)
VALOR = 220
R_RECORTE = 40     # recorte de 81 x 81
LEJOS = 15         # no puede haber lesión ni marca a menos de 15 px de la cruz

# Máscara de la cruz dentro del recorte
cruz = np.zeros((2 * R_RECORTE + 1, 2 * R_RECORTE + 1), dtype=bool)
c = R_RECORTE
cruz[c - MEDIO:c + MEDIO + 1, c - GROSOR:c + GROSOR + 1] = True   # brazo vertical
cruz[c - GROSOR:c + GROSOR + 1, c - MEDIO:c + MEDIO + 1] = True   # brazo horizontal

rng = np.random.default_rng(0)
cache = {}   # para no recalcular imagen, lesión y marcas de la misma imagen
brillo, detectada, psnr_telea, psnr_bih = [], [], [], []

while len(brillo) < N_CRUCES:
    i = rng.integers(len(df))
    if i not in cache:
        img, lesion = cargar(df.iloc[i])
        cache[i] = (img, lesion, detectar_marcas(img))
    img, lesion, marcas_reales = cache[i]
    alto, ancho = img.shape
    if alto <= 2 * R_RECORTE or ancho <= 2 * R_RECORTE:
        continue
    y = rng.integers(R_RECORTE, alto - R_RECORTE)
    x = rng.integers(R_RECORTE, ancho - R_RECORTE)

    # Zona alrededor de la cruz: sin lesión, sin marcas y sin fondo negro
    zona = (slice(y - MEDIO - LEJOS, y + MEDIO + LEJOS + 1), slice(x - MEDIO - LEJOS, x + MEDIO + LEJOS + 1))
    if lesion[zona].any() or marcas_reales[zona].any() or (img[zona] == 0).mean() > 0.05:
        continue

    original = img[y - R_RECORTE:y + R_RECORTE + 1, x - R_RECORTE:x + R_RECORTE + 1].copy()
    con_cruz = original.copy()
    con_cruz[cruz] = VALOR

    # a) Detección
    detectado = detectar_marcas(con_cruz)
    detectada.append(detectado[cruz].all())
    brillo.append(original[cruz].mean())

    # b) Relleno con la máscara verdadera
    for metodo, lista in [(rellenar_telea, psnr_telea), (rellenar_biharmonico, psnr_bih)]:
        relleno = metodo(con_cruz, cruz)
        mse = np.mean((relleno[cruz].astype(float) - original[cruz].astype(float)) ** 2)
        lista.append(10 * np.log10(255 ** 2 / max(mse, 1e-10)))

brillo = np.array(brillo)
detectada = np.array(detectada)
psnr_telea = np.array(psnr_telea)
psnr_bih = np.array(psnr_bih)

# a) Detección por rango de brillo
print(f"a) Detección: {detectada.sum()} de {N_CRUCES} cruces detectadas completas "
      f"({numero(100 * detectada.mean(), 1)} %)")
print("\n   Brillo del tejido   n    % detectadas")
rangos = np.arange(0, 256, 20)
etiquetas, porcentajes, ns = [], [], []
for desde in rangos:
    en_rango = (brillo >= desde) & (brillo < desde + 20)
    if en_rango.sum() == 0:
        continue
    pct = 100 * detectada[en_rango].mean()
    etiquetas.append(f"{desde}–{desde + 19}")
    porcentajes.append(pct)
    ns.append(en_rango.sum())
    print(f"   {desde:3d}–{desde + 19:3d}          {en_rango.sum():4d}   {numero(pct, 1):>6s} %")

fig, eje = plt.subplots(figsize=(9, 4.5))
barras = eje.bar(etiquetas, porcentajes, color="tab:blue")
for barra, n in zip(barras, ns):
    eje.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 1, f"n={n}",
             ha="center", fontsize=8)
eje.set_ylim(0, 110)
eje.set_xlabel("Brillo medio del tejido debajo de la cruz (nivel de gris)")
eje.set_ylabel("% de cruces detectadas completas")
eje.set_title(f"Detección de cruces sintéticas (valor {VALOR}) según el tejido de abajo")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "04a_deteccion_por_brillo.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"   Figura guardada en {ruta_figura}")

# b) Relleno: PSNR
estadistico, p = wilcoxon(psnr_telea, psnr_bih)
print("\nb) Relleno (PSNR dentro de la cruz, en dB; más alto = mejor)")
print("   Test: Wilcoxon de rangos con signo, pareado (mismas cruces)")
print(f"   n = {len(psnr_telea)}")
print(f"   Mediana Telea       = {numero(np.median(psnr_telea), 2)} dB")
print(f"   Mediana biharmónico = {numero(np.median(psnr_bih), 2)} dB")
print(f"   Biharmónico mejor que Telea en {(psnr_bih > psnr_telea).sum()} de {len(psnr_telea)} cruces")
print(f"   W = {numero(estadistico, 1)}   {formato_p(p)}")

ganador = "biharmonico" if np.median(psnr_bih) > np.median(psnr_telea) else "telea"
print(f"\n   Método ganador: {ganador}")
if ganador != marcas.METODO:
    print(f"   ATENCIÓN: marcas.py usa METODO = \"{marcas.METODO}\". Cambiarlo a \"{ganador}\".")


# ---------------------------------------------------------------------------
# BLOQUE 5. Aplicar limpiar(img) a las 780 imágenes.
# limpiar (en marcas.py) detecta las marcas y las rellena con el método ganador.
# Imprimimos qué porcentaje de cada imagen ocupan las marcas (mediana).
# ---------------------------------------------------------------------------
titulo("BLOQUE 5. limpiar() sobre las 780 imágenes")

porcentaje_marcas = []
for _, fila in df.iterrows():
    img, _ = cargar(fila)
    imagen_limpia, mascara = limpiar(img)
    porcentaje_marcas.append(100 * mascara.mean())
df["pct_marcas"] = porcentaje_marcas

print(f"Método usado por limpiar(): {marcas.METODO}")
print(f"Mediana del % de la imagen ocupado por marcas: {numero(df['pct_marcas'].median(), 2)} %")
for clase in CLASES:
    print(f"  {clase:10s} {numero(df.loc[df['clase'] == clase, 'pct_marcas'].median(), 2)} %")


# ---------------------------------------------------------------------------
# RESUMEN
# ---------------------------------------------------------------------------
titulo("RESUMEN")

mejor, peor = (psnr_bih, psnr_telea) if ganador == "biharmonico" else (psnr_telea, psnr_bih)
print(f"- Método elegido: {ganador}, porque su PSNR mediano es mayor "
      f"({numero(np.median(mejor), 2)} vs. {numero(np.median(peor), 2)} dB, {formato_p(p)}).")
print(f"- Detección de cruces sintéticas: {numero(100 * detectada.mean(), 1)} % completas.")
print(f"- PSNR mediano: Telea {numero(np.median(psnr_telea), 2)} dB, "
      f"biharmónico {numero(np.median(psnr_bih), 2)} dB.")
print("- Limitaciones:")
print("  * Marcas sobre tejido muy brillante: sobresalen menos de 100 y no se detectan.")
print("  * Marcas pegadas a fibras: se unen en un solo objeto largo y horizontal,")
print("    que puede descartarse como fibra.")
print("  * Las 16 imágenes con color: se pasan a gris y las marcas de color pueden")
print("    no quedar claras, así que el top-hat no las ve.")
