# paso3_speckle.py
# Caracterizar el ruido de las ecografías (speckle): el granulado que aparece
# aun en tejido homogéneo. Lo medimos en parches de tejido sin lesión ni marcas.
# Se corre desde la carpeta del proyecto:  python3 paso3_speckle.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso3/.

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import gamma

from comun import armar_tabla, cargar
from marcas import limpiar

CARPETA_SALIDA = "outputs/paso3"
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


# ---------------------------------------------------------------------------
# BLOQUE 1. Parches de tejido.
# Para medir el ruido necesitamos zonas donde el tejido sea "parejo": así,
# todo lo que varía dentro del parche es speckle y no estructura (bordes,
# fibras, lesión). Tomamos 5 parches de 32x32 px por imagen, en posiciones
# al azar (semilla 13) que no se superponen, sobre las imágenes limpias.
# Se descartan, en este orden (cada parche se cuenta en la primera regla que falla):
#   a) a menos de 20 px de la lesión (el borde de la lesión no es tejido parejo);
#   b) tocan píxeles de marcas (aunque estén rellenas, ahí el valor es inventado);
#   c) fondo negro (media < 20, desvío < 3 o algún píxel = 0) o saturados
#      (algún píxel >= 250): ahí el ruido está recortado y no se puede medir;
#   d) no homogéneos: se divide el parche en 4 cuadrantes de 16x16; si la media
#      mayor menos la menor supera un desvío del parche, hay un cambio de tejido.
# ---------------------------------------------------------------------------
titulo("BLOQUE 1. Parches de tejido")

TAM = 32
POR_IMAGEN = 5
DIST_LESION = 20

rng = np.random.default_rng(13)
elegidos = []            # parches que pasan todas las reglas
posiciones_elegidas = [] # (id, y, x) de cada parche elegido, para usarlos en el paso 4
no_homogeneos = []       # parches descartados por la regla d (para la figura)
descartes = {"a) cerca de la lesión": 0, "b) tocan marcas": 0,
             "c) fondo negro o saturado": 0, "d) no homogéneos": 0}
total = 0

for _, fila in df.iterrows():
    img, lesion = cargar(fila)
    limpia, mascara_marcas = limpiar(img)
    # Distancia de cada píxel a la lesión (si no hay lesión, todo queda "lejos")
    if lesion.any():
        distancia = ndimage.distance_transform_edt(~lesion)
    else:
        distancia = np.full(img.shape, np.inf)

    # Elegir 5 posiciones al azar que no se superpongan
    alto, ancho = img.shape
    posiciones = []
    intentos = 0
    while len(posiciones) < POR_IMAGEN and intentos < 1000:
        intentos += 1
        y = rng.integers(0, alto - TAM + 1)
        x = rng.integers(0, ancho - TAM + 1)
        se_superpone = any(abs(y - y2) < TAM and abs(x - x2) < TAM for y2, x2 in posiciones)
        if not se_superpone:
            posiciones.append((y, x))

    for y, x in posiciones:
        total += 1
        zona = (slice(y, y + TAM), slice(x, x + TAM))
        parche = limpia[zona].astype(float)
        media, desvio = parche.mean(), parche.std()

        if distancia[zona].min() < DIST_LESION:
            descartes["a) cerca de la lesión"] += 1
            continue
        if mascara_marcas[zona].any():
            descartes["b) tocan marcas"] += 1
            continue
        if media < 20 or desvio < 3 or (parche == 0).any() or (parche >= 250).any():
            descartes["c) fondo negro o saturado"] += 1
            continue
        medias_cuadrantes = [parche[:16, :16].mean(), parche[:16, 16:].mean(),
                             parche[16:, :16].mean(), parche[16:, 16:].mean()]
        if max(medias_cuadrantes) - min(medias_cuadrantes) > desvio:
            descartes["d) no homogéneos"] += 1
            no_homogeneos.append(parche)
            continue
        elegidos.append(parche)
        posiciones_elegidas.append((fila["id"], y, x))

print(f"Parches sorteados: {total}")
for regla, n in descartes.items():
    print(f"  Descartados por {regla}: {n}")
print(f"Parches que quedaron: {len(elegidos)}")

# Guardamos dónde quedó cada parche (los usa paso4_filtros.py)
ruta_csv = os.path.join(CARPETA_SALIDA, "parches.csv")
pd.DataFrame(posiciones_elegidas, columns=["id", "y", "x"]).to_csv(ruta_csv, index=False)
print(f"Posiciones guardadas en {ruta_csv}")

# Figura: 8 parches elegidos (arriba) y 8 descartados por no ser homogéneos (abajo)
fig, ejes = plt.subplots(2, 8, figsize=(16, 4.6))
for j in range(8):
    for i, (lista, nombre) in enumerate([(elegidos, "elegido"), (no_homogeneos, "no homogéneo")]):
        parche = lista[j * len(lista) // 8]   # repartidos a lo largo de la lista
        ejes[i, j].imshow(parche, cmap="gray", vmin=0, vmax=255)
        ejes[i, j].set_title(f"{nombre}\nmedia {parche.mean():.0f}", fontsize=8)
        ejes[i, j].axis("off")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "01_parches.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"Figura guardada en {ruta_figura}")

medias = np.array([p.mean() for p in elegidos])
desvios = np.array([p.std() for p in elegidos])


# ---------------------------------------------------------------------------
# BLOQUE 2. Desvío vs. media.
# Si el ecógrafo guardara la amplitud tal cual (speckle de Rayleigh), el desvío
# sería proporcional a la media: desvío = media / 1,91 (la línea del gráfico).
# Pero las ecografías se guardan con compresión logarítmica, que achica las
# diferencias en lo brillante: el desvío deja de crecer con la media y tiende
# a un valor fijo. Ese valor lo medimos en los parches brillantes (media > 110).
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Desvío vs. media")

brillantes = medias > 110
desvio_brillantes = np.median(desvios[brillantes])

fig, eje = plt.subplots(figsize=(7, 5))
eje.scatter(medias, desvios, s=6, alpha=0.4, color="tab:blue", label="parches")
x_linea = np.linspace(0, medias.max(), 100)
eje.plot(x_linea, x_linea / 1.91, color="tab:red", label="sin compresión (Rayleigh): desvío = media / 1,91")
eje.axhline(desvio_brillantes, color="black", linestyle="--",
            label="con compresión logarítmica: desvío constante")
eje.set_xlabel("Media del parche (nivel de gris)")
eje.set_ylabel("Desvío del parche")
eje.legend()
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "02_desvio_vs_media.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"Figura guardada en {ruta_figura}")

print(f"Parches con media > 110: {brillantes.sum()}")
print(f"Mediana del desvío de esos parches: {numero(desvio_brillantes, 2)}")


# ---------------------------------------------------------------------------
# BLOQUE 3. Descomprimir y ajustar Gamma a la intensidad.
# Si gris = D · ln(amplitud), con amplitud de Rayleigh, el desvío del gris es
# D · 0,64 (0,64 = π / raíz(24), el desvío del logaritmo de una Rayleigh).
# Entonces D = desvío de los parches brillantes / 0,64, y podemos volver a la
# amplitud con amplitud = exp(gris / D), y a la intensidad = amplitud².
# Después ajustamos una Gamma a la intensidad de cada parche.
# Teoría: con speckle puro (muchos dispersores chiquitos y desordenados) la
# amplitud es Rayleigh y la intensidad es exponencial, que es una Gamma con
# forma α = 1. Entonces:
#   - α = 1: speckle puro;
#   - α < 1: más variable que el speckle puro (pocos dispersores, o mezcla de tejidos);
#   - α > 1: más pareja (hay alguna estructura que domina, o el ruido quedó achicado).
# Separamos parches claros (media > 100) y oscuros, porque en el bloque 2 se
# vio que el desvío se comporta distinto en cada zona.
# ---------------------------------------------------------------------------
titulo("BLOQUE 3. Descompresión y ajuste de Gamma a la intensidad")

D = desvio_brillantes / 0.64
print(f"D = {numero(desvio_brillantes, 2)} / 0,64 = {numero(D, 2)}")

valores_alfa = []
for parche in elegidos:
    amplitud = np.exp(parche / D).ravel()
    intensidad = amplitud ** 2
    alfa, _, _ = gamma.fit(intensidad, floc=0)
    valores_alfa.append(alfa)
valores_alfa = np.array(valores_alfa)

print(f"Parches ajustados: {len(valores_alfa)}")
claros = medias > 100
for nombre, grupo in [("Claros (media > 100)", claros), ("Oscuros (media <= 100)", ~claros)]:
    alfas = valores_alfa[grupo]
    cerca_de_1 = (alfas >= 0.8) & (alfas <= 1.2)
    print(f"  {nombre}: n = {grupo.sum()}")
    print(f"    Mediana de α: {numero(np.median(alfas), 2)}")
    print(f"    Parches con α entre 0,8 y 1,2: {numero(100 * cerca_de_1.mean(), 1)} %")

fig, eje = plt.subplots(figsize=(7, 4.5))
eje.hist(valores_alfa, bins=60, color="tab:blue")
eje.axvline(1, color="tab:red", label="α = 1 (speckle puro)")
eje.set_xlabel("α de la Gamma (intensidad)")
eje.set_ylabel("Cantidad de parches")
eje.legend()
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "03_histograma_alfa.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"Figura guardada en {ruta_figura}")


# ---------------------------------------------------------------------------
# BLOQUE 4. Varianza del ruido.
# El filtro de Lee necesita saber cuánta varianza aporta el ruido en una zona
# pareja, para distinguirla de la variación por estructura real.
# Usamos la mediana de la varianza de los parches homogéneos (en niveles de gris
# de la imagen limpia, que es donde se va a aplicar el filtro) y la guardamos.
# ---------------------------------------------------------------------------
titulo("BLOQUE 4. Varianza del ruido")

varianza_ruido = np.median(desvios ** 2)
print(f"Mediana de la varianza de los {len(elegidos)} parches homogéneos: {numero(varianza_ruido, 2)}")

ruta_txt = os.path.join(CARPETA_SALIDA, "varianza_ruido.txt")
with open(ruta_txt, "w") as archivo:
    archivo.write(f"{varianza_ruido:.4f}\n")
print(f"Guardada en {ruta_txt}")


# ---------------------------------------------------------------------------
# BLOQUE 5. Varianza local por imagen.
# En vez de parches sueltos, miramos toda la imagen: para cada píxel calculamos
# la varianza en una ventana de 7x7 a su alrededor, y nos quedamos con la
# mediana de cada imagen. Así vemos cuánto cambia el "nivel de variación"
# de una imagen a otra.
#   - Varianza local = media de x² − (media de x)², con las medias en la ventana.
#   - Solo usamos ventanas que caen enteras dentro del abanico de la sonda y no
#     tocan marcas. El abanico = lo que no es fondo negro (gris > 0), con los
#     huecos rellenados, porque dentro del tejido oscuro también hay píxeles
#     sueltos que valen 0 y no son fondo.
# ---------------------------------------------------------------------------
titulo("BLOQUE 5. Varianza local (ventanas de 7x7) por imagen")

VENTANA = 7
medianas_varianza = []
for _, fila in df.iterrows():
    img, _ = cargar(fila)
    limpia, mascara_marcas = limpiar(img)
    x = limpia.astype(float)

    media_local = ndimage.uniform_filter(x, VENTANA)
    media_cuadrados = ndimage.uniform_filter(x ** 2, VENTANA)
    varianza_local = media_cuadrados - media_local ** 2

    abanico = ndimage.binary_fill_holes(ndimage.binary_closing(limpia > 0, np.ones((5, 5))))
    valido = abanico & ~mascara_marcas
    # La ventana entera tiene que ser válida: erosionamos con un cuadrado de 7x7
    valido = ndimage.binary_erosion(valido, np.ones((VENTANA, VENTANA)))

    medianas_varianza.append(np.median(varianza_local[valido]))
medianas_varianza = np.array(medianas_varianza)

print(f"Mediana de la varianza local en cada una de las {len(medianas_varianza)} imágenes:")
print(f"  Mínimo:       {numero(medianas_varianza.min(), 1)}")
print(f"  Percentil 25: {numero(np.percentile(medianas_varianza, 25), 1)}")
print(f"  Percentil 50: {numero(np.percentile(medianas_varianza, 50), 1)}")
print(f"  Percentil 75: {numero(np.percentile(medianas_varianza, 75), 1)}")
print(f"  Máximo:       {numero(medianas_varianza.max(), 1)}")

fig, eje = plt.subplots(figsize=(7, 4.5))
eje.hist(medianas_varianza, bins=50, color="tab:blue")
eje.set_xlabel("Mediana de la varianza local 7x7 de la imagen")
eje.set_ylabel("Cantidad de imágenes")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "05_varianza_local.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"Figura guardada en {ruta_figura}")
