# paso2_marcas.py
# Detectar y borrar las marcas del ecógrafo (cruces de medición, líneas
# punteadas, números y textos) que vimos en el paso 1, y guardar las imágenes
# limpias para que los pasos siguientes no tengan que volver a calcularlas.
# Se corre desde la carpeta del proyecto:  python3 paso2_marcas.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso2_marcas/.
#
# Necesita haber corrido el paso 1 (usa grupos_repetidas.csv para sacar las
# 19 imágenes con etiqueta contradictoria: trabaja con 761).
#
# Lo lento es rellenar las 761 imágenes. Se hace UNA vez (bloque 0) y queda
# guardado: las imágenes limpias en outputs/paso2_marcas/imagenes_limpias/ y lo
# medido en marcas_por_imagen.csv. Las corridas siguientes leen eso.
# Para rehacerlo (por ejemplo, si cambia algún parámetro de marcas.py), poner
# RECALCULAR = True.
#
# Figuras:
#   01_perfil_y_tophat.png        perfil de grises y top-hat sobre una cruz real
#   02a_tophat_tejido.png         histograma del top-hat en el tejido, con el umbral
#   02b_pasos_deteccion.png       los pasos de la detección sobre un ejemplo
#   03_relleno_ejemplos.png       original, marcas, Telea y biharmónico en 3 ejemplos
#   04a_deteccion_por_brillo.png  % de cruces sintéticas detectadas según el tejido
#   04b_psnr_telea_vs_bih.png     PSNR de los dos métodos en cada cruz sintética
#   04c_ejemplo_cruz.png          una cruz sintética: original, con cruz, rellenos y error
#   05a_pct_marcas.png            % de la imagen ocupado por marcas, por clase
#   05b_antes_despues.png         las imágenes con más marcas, antes y después

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import disk, opening, dilation
from skimage.restoration import inpaint_biharmonic
from scipy.stats import wilcoxon

from comun import CLASES, armar_tabla, cargar, agregar_grupos, ruta_limpia, guardar_limpia, cargar_limpia
import marcas
from marcas import detectar_marcas, rellenar

CARPETA_SALIDA = "outputs/paso2_marcas"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

RECALCULAR = False   # True: vuelve a detectar, rellenar y guardar las 761 imágenes
RUTA_TABLA = os.path.join(CARPETA_SALIDA, "marcas_por_imagen.csv")
RUTA_HIST = os.path.join(CARPETA_SALIDA, "histograma_tophat_tejido.npy")

df = agregar_grupos(armar_tabla())   # 761 imágenes (sin las de etiqueta contradictoria)

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


def guardar_figura(nombre, dpi=120):
    """Guarda la figura actual en la carpeta de salida, la cierra e imprime la ruta."""
    ruta = os.path.join(CARPETA_SALIDA, nombre)
    plt.savefig(ruta, dpi=dpi)
    plt.close()
    print(f"Figura guardada en {ruta}")


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


def rojo_encima(eje, mascara, color=(1, 0, 0)):
    """Pinta la máscara de un color, opaco, encima de lo que ya hay en el eje."""
    capa = np.zeros(mascara.shape + (4,))
    capa[mascara] = list(color) + [1]
    eje.imshow(capa)


# ---------------------------------------------------------------------------
# BLOQUE 0. Recorrido de las 761 imágenes (se hace una sola vez).
# Para cada imagen, en una sola pasada:
#   - detecta las marcas (una vez, y se reusa para todo lo demás);
#   - para el bloque 1: si tiene tejido >= 250 fuera de las marcas;
#   - para el bloque 2: suma el histograma del top-hat en el tejido sin marcas
#     (sin el fondo negro, valor 0, que tiene top-hat 0);
#   - para el bloque 5: qué % de la imagen ocupan las marcas;
#   - rellena las marcas con el método de marcas.py y guarda la imagen limpia.
# ---------------------------------------------------------------------------
titulo(f"BLOQUE 0. Detección, relleno y guardado de las {len(df)} imágenes")

todas_guardadas = all(os.path.exists(ruta_limpia(fila)) for _, fila in df.iterrows())
if not RECALCULAR and todas_guardadas and os.path.exists(RUTA_TABLA) and os.path.exists(RUTA_HIST):
    tabla = pd.read_csv(RUTA_TABLA)
    hist_tejido = np.load(RUTA_HIST)
    print(f"Leído de {RUTA_TABLA} y de las imágenes limpias ya guardadas (no se recalcula).")
else:
    print(f"Método de relleno: {marcas.METODO}. Tarda varios minutos (solo la primera vez).")
    filas_tabla = []
    hist_tejido = np.zeros(256)
    for k, (_, fila) in enumerate(df.iterrows(), start=1):
        img, _ = cargar(fila)
        mascara = detectar_marcas(img)
        th = tophat_blanco(img)
        tejido = ~dilation(mascara, disk(2)) & (img > 0)
        hist_tejido += np.bincount(th[tejido], minlength=256)
        guardar_limpia(fila, rellenar(img, mascara))
        filas_tabla.append({
            "id": fila["id"], "clase": fila["clase"],
            "pct_marcas": 100 * mascara.mean(),
            "tejido_250": bool(((img >= 250) & ~mascara).any()),
        })
        if k % 100 == 0:
            print(f"  {k} de {len(df)}", flush=True)
    tabla = pd.DataFrame(filas_tabla)
    tabla.to_csv(RUTA_TABLA, index=False)
    np.save(RUTA_HIST, hist_tejido)
    print(f"Imágenes limpias guardadas en outputs/paso2_marcas/imagenes_limpias/")
    print(f"Tabla guardada en {RUTA_TABLA}")

df = df.merge(tabla[["id", "pct_marcas", "tejido_250"]], on="id")


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

print(f"\nImágenes con tejido >= 250 (fuera de las marcas): {df['tejido_250'].sum()} de {len(df)}")
print("  -> Un umbral de brillo marcaría tejido como marca y no vería las marcas de ~220.")


# ---------------------------------------------------------------------------
# BLOQUE 2. Detección con top-hat blanco (función detectar_marcas en marcas.py).
# El top-hat mide cuánto sobresale cada píxel de su entorno inmediato, en vez
# de mirar el brillo absoluto. Una marca de ~220 sobre tejido de ~90 sobresale
# ~130; el tejido, aunque sea brillante, casi nunca sobresale tanto en algo
# tan angosto (3 px). La explicación con números en 1D está en marcas.py.
# Para justificar el umbral 100 imprimimos:
#   - el top-hat sobre la cruz del perfil anterior;
#   - el percentil 99,9 del top-hat en el tejido sin marcas de las 761 imágenes.
# Figuras:
#   01:  el perfil de grises y el top-hat de la misma fila, con el umbral.
#   02a: histograma del top-hat en el tejido (escala log), con el umbral.
#   02b: los pasos de la detección sobre "malignant (12)": original, top-hat,
#        candidatos (top-hat >= 100), marcas (rojo) y fibras descartadas
#        (celeste), y la máscara final agrandada.
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Detección con top-hat blanco")

th12 = tophat_blanco(img12)
print(f'Top-hat en la misma fila de "malignant (12)":')
print("  columna: " + " ".join(f"{x:4d}" for x in range(X_INICIO, X_FIN)))
print("  top-hat: " + " ".join(f"{v:4d}" for v in th12[FILA, X_INICIO:X_FIN]))

acumulado = np.cumsum(hist_tejido) / hist_tejido.sum()
p999 = np.searchsorted(acumulado, 0.999)
print(f"\nPercentil 99,9 del top-hat en el tejido sin marcas ({len(df)} imágenes): {p999}")
print(f"Píxeles de tejido con top-hat >= {marcas.UMBRAL_TOPHAT}: "
      f"{numero(100 * (1 - acumulado[marcas.UMBRAL_TOPHAT - 1]), 3)} %")
print(f"  -> Umbral = {marcas.UMBRAL_TOPHAT}: por encima de casi todo el tejido y por debajo de las marcas.")
print("     Lo poco de tejido que pasa el umbral son fibras finas, que se descartan por forma.")

# 01: perfil de grises y top-hat
columnas = np.arange(X_INICIO, X_FIN)
fig, ejes = plt.subplots(1, 2, figsize=(13, 4))
ejes[0].plot(columnas, img12[FILA, X_INICIO:X_FIN], "o-", color="black")
ejes[0].axhline(255, color="gray", linestyle=":", label="255 (blanco puro)")
ejes[0].set_title(f"Gris en la fila {FILA}: la marca vale ~220, no 255")
ejes[0].set_ylabel("Nivel de gris")
ejes[0].set_ylim(0, 265)
ejes[1].plot(columnas, th12[FILA, X_INICIO:X_FIN], "o-", color="tab:red")
ejes[1].axhline(marcas.UMBRAL_TOPHAT, color="tab:red", linestyle="--", label=f"umbral = {marcas.UMBRAL_TOPHAT}")
ejes[1].set_title("Top-hat en la misma fila: solo la marca sobresale")
ejes[1].set_ylabel("Top-hat (cuánto sobresale de su entorno)")
ejes[1].set_ylim(0, 265)
for eje in ejes:
    eje.set_xlabel("Columna (píxel)")
    eje.legend()
    eje.spines[["top", "right"]].set_visible(False)
plt.suptitle('"malignant (12)": perfil que cruza el brazo vertical de una cruz de medición')
plt.tight_layout()
guardar_figura("01_perfil_y_tophat.png")

# 02a: histograma del top-hat en el tejido
fig, eje = plt.subplots(figsize=(9, 4.5))
eje.bar(np.arange(256), hist_tejido / hist_tejido.sum(), width=1, color="#5a5a5a")
eje.axvline(marcas.UMBRAL_TOPHAT, color="tab:red", linestyle="--", label=f"umbral = {marcas.UMBRAL_TOPHAT}")
eje.axvline(p999, color="tab:blue", linestyle=":", label=f"percentil 99,9 = {p999}")
eje.set_yscale("log")
eje.set_xlabel("Top-hat")
eje.set_ylabel("Proporción de píxeles de tejido (escala log)")
eje.set_title(f"Top-hat en el tejido sin marcas ({len(df)} imágenes): casi nada llega al umbral")
eje.legend()
eje.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
guardar_figura("02a_tophat_tejido.png")

# 02b: pasos de la detección
marcas12, fibras12 = detectar_marcas(img12, devolver_fibras=True)
fig, ejes = plt.subplots(1, 4, figsize=(20, 5))
ejes[0].imshow(img12, cmap="gray", vmin=0, vmax=255)
ejes[0].set_title("1. Original")
ejes[1].imshow(th12, cmap="magma", vmin=0, vmax=200)
ejes[1].set_title("2. Top-hat (claro = sobresale de su entorno)")
ejes[2].imshow(img12, cmap="gray", vmin=0, vmax=255)
rojo_encima(ejes[2], th12 >= marcas.UMBRAL_TOPHAT, (1, 1, 0))
ejes[2].set_title(f"3. Candidatos: top-hat >= {marcas.UMBRAL_TOPHAT} (amarillo)")
ejes[3].imshow(img12, cmap="gray", vmin=0, vmax=255)
rojo_encima(ejes[3], marcas12)
rojo_encima(ejes[3], fibras12, (0, 0.8, 1))
ejes[3].set_title(f"4. Marcas finales (rojo, agrandadas {marcas.AGRANDAR} px)\ny fibras descartadas (celeste)")
for eje in ejes:
    eje.axis("off")
plt.tight_layout()
guardar_figura("02b_pasos_deteccion.png", dpi=100)


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
            rojo_encima(ejes[i, j], mascara)
        ejes[i, j].set_title(f"{id_imagen} – {nombre}", fontsize=10)
        ejes[i, j].axis("off")
    print(f"  {id_imagen}: las marcas ocupan {numero(100 * mascara.mean(), 2)} % de la imagen")
plt.tight_layout()
guardar_figura("03_relleno_ejemplos.png", dpi=100)


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
# Figuras extra:
#   04b: PSNR de Telea contra PSNR de biharmónico, un punto por cruz. Los puntos
#        por encima de la diagonal son cruces donde ganó el biharmónico.
#   04c: una cruz de ejemplo: original, con cruz, los dos rellenos y su error.
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
ejemplo_cruz = None

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
    rellenos = {}
    for nombre, metodo, lista in [("Telea", rellenar_telea, psnr_telea),
                                  ("biharmónico", rellenar_biharmonico, psnr_bih)]:
        relleno = metodo(con_cruz, cruz)
        rellenos[nombre] = relleno
        mse = np.mean((relleno[cruz].astype(float) - original[cruz].astype(float)) ** 2)
        lista.append(10 * np.log10(255 ** 2 / max(mse, 1e-10)))

    # Guardamos la primera cruz sobre tejido de brillo intermedio como ejemplo
    if ejemplo_cruz is None and 80 <= original[cruz].mean() <= 140:
        ejemplo_cruz = (df.iloc[i]["id"], original, con_cruz, detectado, rellenos)

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
guardar_figura("04a_deteccion_por_brillo.png")

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
    print(f"   ATENCIÓN: marcas.py usa METODO = \"{marcas.METODO}\". Cambiarlo a \"{ganador}\"")
    print("   y volver a correr este script con RECALCULAR = True.")

# 04b: PSNR de los dos métodos, cruz por cruz
fig, eje = plt.subplots(figsize=(6, 6))
eje.scatter(psnr_telea, psnr_bih, s=10, alpha=0.5, color="#2a78b5", edgecolors="none")
limites = [min(psnr_telea.min(), psnr_bih.min()) - 1, max(psnr_telea.max(), psnr_bih.max()) + 1]
eje.plot(limites, limites, "k--", linewidth=1, label="iguales")
eje.set_xlim(limites)
eje.set_ylim(limites)
eje.set_aspect("equal")
eje.set_xlabel("PSNR Telea (dB)")
eje.set_ylabel("PSNR biharmónico (dB)")
eje.set_title(f"Arriba de la diagonal gana el biharmónico\n"
              f"({(psnr_bih > psnr_telea).sum()} de {len(psnr_telea)} cruces, Wilcoxon {formato_p(p)})")
eje.legend(loc="lower right")
eje.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
guardar_figura("04b_psnr_telea_vs_bih.png")

# 04c: una cruz de ejemplo
if ejemplo_cruz is not None:
    id_ej, original, con_cruz, detectado, rellenos = ejemplo_cruz
    fig, ejes = plt.subplots(2, 4, figsize=(16, 8.5))
    paneles = [(original, "original (lo que había)"), (con_cruz, f"con la cruz ({VALOR})")]
    for j, (imagen, nombre) in enumerate(paneles):
        ejes[0, j].imshow(imagen, cmap="gray", vmin=0, vmax=255)
        ejes[0, j].set_title(nombre)
    ejes[0, 2].imshow(con_cruz, cmap="gray", vmin=0, vmax=255)
    rojo_encima(ejes[0, 2], detectado)
    ejes[0, 2].set_title("detectado (rojo)")
    ejes[0, 3].axis("off")
    for j, nombre in enumerate(["Telea", "biharmónico"]):
        relleno = rellenos[nombre]
        error = np.abs(relleno.astype(int) - original.astype(int))
        mse = np.mean(error[cruz].astype(float) ** 2)
        ejes[1, 2 * j].imshow(relleno, cmap="gray", vmin=0, vmax=255)
        ejes[1, 2 * j].set_title(f"relleno {nombre}")
        ejes[1, 2 * j + 1].imshow(error, cmap="magma", vmin=0, vmax=60)
        ejes[1, 2 * j + 1].set_title(f"|error| {nombre}: PSNR = {numero(10 * np.log10(255 ** 2 / max(mse, 1e-10)), 1)} dB")
    for eje in ejes.ravel():
        eje.set_xticks([])
        eje.set_yticks([])
    plt.suptitle(f"Cruz sintética sobre {id_ej} (recorte de {2 * R_RECORTE + 1} × {2 * R_RECORTE + 1} px; "
                 f"error: negro = 0, claro = 60 o más)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    guardar_figura("04c_ejemplo_cruz.png")


# ---------------------------------------------------------------------------
# BLOQUE 5. Resultado sobre las 761 imágenes.
# Las imágenes ya se limpiaron y guardaron en el bloque 0, con el método de
# marcas.py. Acá se resume cuánto de cada imagen ocupaban las marcas.
# Figuras:
#   05a: % de la imagen ocupado por marcas, por clase.
#   05b: las 4 imágenes con más marcas, antes y después de limpiarlas.
# ---------------------------------------------------------------------------
titulo(f"BLOQUE 5. Marcas en las {len(df)} imágenes")

print(f"Método usado: {marcas.METODO}")
print(f"Imágenes sin ninguna marca detectada: {(df['pct_marcas'] == 0).sum()} de {len(df)}")
print(f"Mediana del % de la imagen ocupado por marcas: {numero(df['pct_marcas'].median(), 2)} %")
for clase in CLASES:
    print(f"  {clase:10s} {numero(df.loc[df['clase'] == clase, 'pct_marcas'].median(), 2)} %")

# 05a: % de marcas por clase
rng_puntos = np.random.default_rng(0)
fig, eje = plt.subplots(figsize=(7, 5))
valores = [df.loc[df["clase"] == c, "pct_marcas"].to_numpy() for c in CLASES]
eje.boxplot(valores, widths=0.5, showfliers=False, medianprops={"color": "black", "linewidth": 2})
for k, (clase, v) in enumerate(zip(CLASES, valores), start=1):
    eje.scatter(k + rng_puntos.uniform(-0.18, 0.18, len(v)), v, s=8, alpha=0.45,
                color=COLORES[clase], edgecolors="none")
eje.set_xticks(range(1, len(CLASES) + 1), [f"{c}\n(n = {len(v)})" for c, v in zip(CLASES, valores)])
eje.set_ylabel("% de la imagen ocupado por marcas")
eje.set_title("Cuánto de cada imagen se rellenó")
eje.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
guardar_figura("05a_pct_marcas.png")

# 05b: antes y después en las imágenes con más marcas
con_mas = df.sort_values("pct_marcas", ascending=False).head(4)
fig, ejes = plt.subplots(2, 4, figsize=(18, 9))
for j, (_, fila) in enumerate(con_mas.iterrows()):
    antes, _ = cargar(fila)
    despues, _ = cargar_limpia(fila)
    ejes[0, j].imshow(antes, cmap="gray", vmin=0, vmax=255)
    ejes[0, j].set_title(f"{fila['id']}: antes ({numero(fila['pct_marcas'], 1)} % marcas)", fontsize=10)
    ejes[1, j].imshow(despues, cmap="gray", vmin=0, vmax=255)
    ejes[1, j].set_title(f"{fila['id']}: después", fontsize=10)
for eje in ejes.ravel():
    eje.axis("off")
plt.suptitle("Las 4 imágenes con más marcas, antes (arriba) y después (abajo) de limpiarlas")
plt.tight_layout()
guardar_figura("05b_antes_despues.png", dpi=100)


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
print(f"- Imágenes limpias guardadas: {len(df)} (los pasos siguientes las leen con cargar_limpia).")
print("- Limitaciones:")
print("  * Marcas sobre tejido muy brillante: sobresalen menos de 100 y no se detectan.")
print("  * Marcas pegadas a fibras: se unen en un solo objeto largo y horizontal,")
print("    que puede descartarse como fibra.")
print("  * Las 16 imágenes con color: se pasan a gris y las marcas de color pueden")
print("    no quedar claras, así que el top-hat no las ve.")