# paso3_speckle.py
# Caracterizar el ruido de las ecografías (speckle): el granulado que aparece
# aun en tejido homogéneo. Lo medimos en parches de tejido sin lesión ni marcas.
# Se corre desde la carpeta del proyecto:  python3 paso3_speckle.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso3_speckle/.
#
# Necesita haber corrido los pasos 1 y 2: usa las 761 imágenes válidas y las
# imágenes limpias que guardó el paso 2 (no vuelve a calcular el relleno).
#
# Figuras:
#   01_parches.png                  parches elegidos y parches descartados por no homogéneos
#   01b_parches_en_imagen.png       dónde cayeron los parches en una imagen, y por qué se descartaron
#   01c_descartes.png               cuántos parches se descartaron por cada regla
#   02_desvio_vs_media.png          desvío contra media de cada parche
#   03_histograma_alfa.png          α de la Gamma, parches claros y oscuros
#   03b_ajuste_gamma_ejemplo.png    un parche: histograma de la intensidad con la Gamma ajustada
#   04_varianza_ruido_por_ventana.png  varianza del ruido medida con 32x32 y con 7x7
#   05_varianza_local.png           mediana de la varianza local 7x7 de cada imagen
#   05b_mapa_varianza_local.png     mapa de varianza local de una imagen y zonas "solo ruido"

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import gamma, expon, spearmanr

from comun import armar_tabla, agregar_grupos, cargar_limpia

CARPETA_SALIDA = "outputs/paso3_speckle"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

df = agregar_grupos(armar_tabla())   # 761 imágenes (sin las de etiqueta contradictoria)

ID_EJEMPLO = "benign (209)"          # imagen de ejemplo para las figuras 01b y 05b
VENTANA = 7                          # ventana del filtro de Lee (paso 4)


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


def guardar_figura(nombre, dpi=120):
    """Guarda la figura actual en la carpeta de salida, la cierra e imprime la ruta."""
    ruta = os.path.join(CARPETA_SALIDA, nombre)
    plt.savefig(ruta, dpi=dpi)
    plt.close()
    print(f"Figura guardada en {ruta}")


def varianza_local(x, ventana):
    """Varianza en una ventana cuadrada alrededor de cada píxel:
    media de x² − (media de x)², con las medias calculadas en la ventana."""
    x = x.astype(float)
    media = ndimage.uniform_filter(x, ventana)
    media_cuadrados = ndimage.uniform_filter(x ** 2, ventana)
    return np.maximum(media_cuadrados - media ** 2, 0)


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
#
# En la misma pasada por las imágenes se calcula también lo del bloque 5
# (varianza local 7x7 de toda la imagen), para leer cada imagen una sola vez.
# ---------------------------------------------------------------------------
titulo("BLOQUE 1. Parches de tejido")

TAM = 32
POR_IMAGEN = 5
DIST_LESION = 20
MUESTRA_POR_IMAGEN = 2000   # valores de varianza local guardados por imagen (bloque 5)

rng = np.random.default_rng(13)
rng_muestra = np.random.default_rng(0)
elegidos = []            # parches que pasan todas las reglas
posiciones_elegidas = [] # (id, y, x) de cada parche elegido, para usarlos en el paso 4
no_homogeneos = []       # parches descartados por la regla d (para la figura)
descartes = {"a) cerca de la lesión": 0, "b) tocan marcas": 0,
             "c) fondo negro o saturado": 0, "d) no homogéneos": 0}
total = 0
ejemplo = None           # imagen de ejemplo y sus parches, para las figuras 01b y 05b
medianas_varianza = []   # bloque 5: mediana de la varianza local de cada imagen
muestras_varianza = []   # bloque 5: muestra de valores de varianza local de cada imagen

for _, fila in df.iterrows():
    limpia, lesion, mascara_marcas = cargar_limpia(fila, con_marcas=True)
    # Distancia de cada píxel a la lesión (si no hay lesión, todo queda "lejos")
    if lesion.any():
        distancia = ndimage.distance_transform_edt(~lesion)
    else:
        distancia = np.full(limpia.shape, np.inf)

    # Elegir 5 posiciones al azar que no se superpongan
    alto, ancho = limpia.shape
    posiciones = []
    intentos = 0
    while len(posiciones) < POR_IMAGEN and intentos < 1000:
        intentos += 1
        y = rng.integers(0, alto - TAM + 1)
        x = rng.integers(0, ancho - TAM + 1)
        se_superpone = any(abs(y - y2) < TAM and abs(x - x2) < TAM for y2, x2 in posiciones)
        if not se_superpone:
            posiciones.append((y, x))

    resultado_parches = []   # (y, x, motivo) de esta imagen, para la figura 01b
    for y, x in posiciones:
        total += 1
        zona = (slice(y, y + TAM), slice(x, x + TAM))
        parche = limpia[zona].astype(float)
        media, desvio = parche.mean(), parche.std()

        if distancia[zona].min() < DIST_LESION:
            motivo = "a) cerca de la lesión"
        elif mascara_marcas[zona].any():
            motivo = "b) tocan marcas"
        elif media < 20 or desvio < 3 or (parche == 0).any() or (parche >= 250).any():
            motivo = "c) fondo negro o saturado"
        else:
            medias_cuadrantes = [parche[:16, :16].mean(), parche[:16, 16:].mean(),
                                 parche[16:, :16].mean(), parche[16:, 16:].mean()]
            if max(medias_cuadrantes) - min(medias_cuadrantes) > desvio:
                motivo = "d) no homogéneos"
                no_homogeneos.append(parche)
            else:
                motivo = None
        resultado_parches.append((y, x, motivo))
        if motivo is not None:
            descartes[motivo] += 1
            continue
        elegidos.append(parche)
        posiciones_elegidas.append((fila["id"], y, x))

    # Bloque 5: varianza local 7x7 en toda la imagen.
    # Solo usamos ventanas que caen enteras dentro del abanico de la sonda y no
    # tocan marcas. El abanico = lo que no es fondo negro (gris > 0), con los
    # huecos rellenados, porque dentro del tejido oscuro también hay píxeles
    # sueltos que valen 0 y no son fondo.
    var_local = varianza_local(limpia, VENTANA)
    abanico = ndimage.binary_fill_holes(ndimage.binary_closing(limpia > 0, np.ones((5, 5))))
    valido = ndimage.binary_erosion(abanico & ~mascara_marcas, np.ones((VENTANA, VENTANA)))
    valores = var_local[valido]
    medianas_varianza.append(np.median(valores))
    muestras_varianza.append(rng_muestra.choice(valores, size=min(MUESTRA_POR_IMAGEN, len(valores)), replace=False))

    if fila["id"] == ID_EJEMPLO:
        ejemplo = (limpia, lesion, resultado_parches, var_local, valido)

print(f"Imágenes: {len(df)}")
print(f"Parches sorteados: {total}")
for regla, n in descartes.items():
    print(f"  Descartados por {regla}: {n}")
print(f"Parches que quedaron: {len(elegidos)}")

# Guardamos dónde quedó cada parche (los usa paso4_filtros.py)
ruta_csv = os.path.join(CARPETA_SALIDA, "parches.csv")
pd.DataFrame(posiciones_elegidas, columns=["id", "y", "x"]).to_csv(ruta_csv, index=False)
print(f"Posiciones guardadas en {ruta_csv}")

# 01: 8 parches elegidos (arriba) y 8 descartados por no ser homogéneos (abajo)
fig, ejes = plt.subplots(2, 8, figsize=(16, 4.6))
for j in range(8):
    for i, (lista, nombre) in enumerate([(elegidos, "elegido"), (no_homogeneos, "no homogéneo")]):
        parche = lista[j * len(lista) // 8]   # repartidos a lo largo de la lista
        ejes[i, j].imshow(parche, cmap="gray", vmin=0, vmax=255)
        ejes[i, j].set_title(f"{nombre}\nmedia {parche.mean():.0f}", fontsize=8)
        ejes[i, j].axis("off")
plt.tight_layout()
guardar_figura("01_parches.png")

# 01b: dónde cayeron los parches en la imagen de ejemplo
COLOR_MOTIVO = {None: "lime", "a) cerca de la lesión": "red", "b) tocan marcas": "orange",
                "c) fondo negro o saturado": "magenta", "d) no homogéneos": "cyan"}
if ejemplo is not None:
    limpia_ej, lesion_ej, parches_ej, _, _ = ejemplo
    fig, eje = plt.subplots(figsize=(9, 7))
    eje.imshow(limpia_ej, cmap="gray", vmin=0, vmax=255)
    if lesion_ej.any():
        eje.contour(lesion_ej, levels=[0.5], colors="yellow", linewidths=1.5)
    usados = set()
    for y, x, motivo in parches_ej:
        etiqueta = "elegido" if motivo is None else motivo
        eje.add_patch(plt.Rectangle((x, y), TAM, TAM, fill=False, linewidth=2,
                                    edgecolor=COLOR_MOTIVO[motivo],
                                    label=None if etiqueta in usados else etiqueta))
        usados.add(etiqueta)
    eje.legend(loc="lower right", fontsize=9)
    eje.set_title(f"{ID_EJEMPLO}: los {len(parches_ej)} parches sorteados (amarillo = lesión)")
    eje.axis("off")
    plt.tight_layout()
    guardar_figura("01b_parches_en_imagen.png")

# 01c: descartes por regla
fig, eje = plt.subplots(figsize=(8, 4))
nombres = ["elegidos"] + list(descartes.keys())
cantidades = [len(elegidos)] + list(descartes.values())
colores = ["#2e9e4f"] + ["#8c8c8c"] * len(descartes)
barras = eje.barh(nombres[::-1], cantidades[::-1], color=colores[::-1])
for barra, n in zip(barras, cantidades[::-1]):
    eje.text(barra.get_width() + total * 0.005, barra.get_y() + barra.get_height() / 2,
             f"{n} ({numero(100 * n / total, 1)} %)", va="center", fontsize=9)
eje.set_xlim(0, max(cantidades) * 1.25)
eje.set_xlabel("Cantidad de parches")
eje.set_title(f"Qué pasó con los {total} parches sorteados")
eje.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
guardar_figura("01c_descartes.png")

medias = np.array([p.mean() for p in elegidos])
desvios = np.array([p.std() for p in elegidos])


# ---------------------------------------------------------------------------
# BLOQUE 2. Desvío vs. media.
# Si el ecógrafo guardara la amplitud tal cual (speckle de Rayleigh), el desvío
# sería proporcional a la media: desvío = media / 1,91 (la línea del gráfico).
# Pero las ecografías se guardan con compresión logarítmica, que achica las
# diferencias en lo brillante: el desvío deja de crecer con la media y tiende
# a un valor fijo. Ese valor lo medimos en los parches brillantes (media > 110).
# En la figura se agrega la mediana del desvío por tramos de media (línea negra
# con puntos), para ver la tendencia sin que la tapen los puntos sueltos.
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. Desvío vs. media")

brillantes = medias > 110
desvio_brillantes = np.median(desvios[brillantes])

tramos = np.arange(0, medias.max() + 10, 10)
centros, medianas_tramo = [], []
for desde in tramos:
    en_tramo = (medias >= desde) & (medias < desde + 10)
    if en_tramo.sum() >= 10:
        centros.append(desde + 5)
        medianas_tramo.append(np.median(desvios[en_tramo]))

fig, eje = plt.subplots(figsize=(7, 5))
eje.scatter(medias, desvios, s=6, alpha=0.3, color="tab:blue", label="parches")
x_linea = np.linspace(0, medias.max(), 100)
eje.plot(x_linea, x_linea / 1.91, color="tab:red", label="sin compresión (Rayleigh): desvío = media / 1,91")
eje.axhline(desvio_brillantes, color="black", linestyle="--",
            label="con compresión logarítmica: desvío constante")
eje.plot(centros, medianas_tramo, "o-", color="black", markersize=4, label="mediana por tramos de 10")
eje.set_xlabel("Media del parche (nivel de gris)")
eje.set_ylabel("Desvío del parche")
eje.set_ylim(0, max(desvios.max(), desvio_brillantes) * 1.1)
eje.legend(fontsize=8)
plt.tight_layout()
guardar_figura("02_desvio_vs_media.png")

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
# Figura extra (03b): un parche claro con α cerca de la mediana, con el
# histograma de su intensidad, la Gamma ajustada y la exponencial (α = 1).
# ---------------------------------------------------------------------------
titulo("BLOQUE 3. Descompresión y ajuste de Gamma a la intensidad")

D = desvio_brillantes / 0.64
print(f"D = {numero(desvio_brillantes, 2)} / 0,64 = {numero(D, 2)}")

valores_alfa, escalas = [], []
for parche in elegidos:
    amplitud = np.exp(parche / D).ravel()
    intensidad = amplitud ** 2
    alfa, _, escala = gamma.fit(intensidad, floc=0)
    valores_alfa.append(alfa)
    escalas.append(escala)
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
bordes = np.linspace(0, np.percentile(valores_alfa, 99), 60)
eje.hist(valores_alfa[claros], bins=bordes, color="#e0a030", alpha=0.7, label=f"claros (media > 100), n = {claros.sum()}")
eje.hist(valores_alfa[~claros], bins=bordes, color="#2a78b5", alpha=0.6, label=f"oscuros, n = {(~claros).sum()}")
eje.axvline(1, color="tab:red", label="α = 1 (speckle puro)")
eje.set_xlabel("α de la Gamma (intensidad)")
eje.set_ylabel("Cantidad de parches")
eje.legend()
plt.tight_layout()
guardar_figura("03_histograma_alfa.png")

# 03b: un parche claro típico, con su ajuste
indices_claros = np.flatnonzero(claros)
if len(indices_claros) > 0:
    mediana_claros = np.median(valores_alfa[claros])
    k = indices_claros[np.argmin(np.abs(valores_alfa[claros] - mediana_claros))]
    intensidad = np.exp(elegidos[k] / D).ravel() ** 2
    escala_media = intensidad.mean()          # se divide por la media para que el eje quede en ~0 a 5
    intensidad = intensidad / escala_media
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 2.2]})
    ejes[0].imshow(elegidos[k], cmap="gray", vmin=0, vmax=255)
    ejes[0].set_title(f"Parche de ejemplo (media {numero(medias[k], 0)})")
    ejes[0].axis("off")
    x_pdf = np.linspace(0, np.percentile(intensidad, 99.5), 300)
    ejes[1].hist(intensidad, bins=50, range=(0, x_pdf[-1]), density=True, color="#bbbbbb", label="intensidad del parche")
    ejes[1].plot(x_pdf, gamma.pdf(x_pdf, valores_alfa[k], scale=escalas[k] / escala_media), color="black",
                 label=f"Gamma ajustada (α = {numero(valores_alfa[k], 2)})")
    ejes[1].plot(x_pdf, expon.pdf(x_pdf, scale=1), color="tab:red", linestyle="--",
                 label="exponencial (α = 1, speckle puro)")
    ejes[1].set_xlabel("Intensidad / su media   (intensidad = exp(gris / D)²)")
    ejes[1].set_ylabel("Densidad")
    ejes[1].legend()
    ejes[1].spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    guardar_figura("03b_ajuste_gamma_ejemplo.png")


# ---------------------------------------------------------------------------
# BLOQUE 4. Varianza del ruido.
# El filtro de Lee necesita saber cuánta varianza aporta el ruido en una zona
# pareja, para distinguirla de la variación por estructura real. Lee compara
# esa referencia con la varianza local en ventanas de 7x7, así que la
# referencia tiene que medirse con la MISMA ventana.
# Esto importa: el speckle tiene granos de varios píxeles, y una ventana chica
# ve solo una parte de la variación. El mismo ruido mide bastante menos en 7x7
# que en el parche entero de 32x32. Si usáramos el valor de 32x32, Lee creería
# que casi todo es ruido y promediaría también los bordes.
# Por eso se calcula, para cada parche homogéneo, la mediana de su varianza
# local 7x7 (sin los 3 px del borde del parche, donde la ventana se sale), y
# después la mediana entre parches. Se imprimen los dos valores y se guarda
# el de 7x7.
# ---------------------------------------------------------------------------
titulo("BLOQUE 4. Varianza del ruido")

r = VENTANA // 2
varianzas_32 = desvios ** 2
varianzas_7 = np.array([np.median(varianza_local(p, VENTANA)[r:-r, r:-r]) for p in elegidos])
varianza_ruido_32 = np.median(varianzas_32)
varianza_ruido = np.median(varianzas_7)
print(f"Mediana de la varianza de los {len(elegidos)} parches homogéneos:")
print(f"  medida en el parche entero (32x32):         {numero(varianza_ruido_32, 2)}")
print(f"  medida en ventanas de {VENTANA}x{VENTANA} (como Lee):      {numero(varianza_ruido, 2)}")
print(f"  -> El mismo ruido mide {numero(varianza_ruido_32 / varianza_ruido, 1)} veces menos en {VENTANA}x{VENTANA}. "
      f"Se usa el de {VENTANA}x{VENTANA}.")

ruta_txt = os.path.join(CARPETA_SALIDA, "varianza_ruido.txt")
with open(ruta_txt, "w") as archivo:
    archivo.write(f"{varianza_ruido:.4f}\n")
print(f"Guardada en {ruta_txt}")

# ¿Tendría sentido un nivel de ruido distinto para cada imagen?
#   a) Para estimarlo hace falta tejido homogéneo de esa imagen: ¿cuántas
#      imágenes tienen 0, 1 o 2 o más parches homogéneos?
#   b) Correlación intraclase ICC(1), en las imágenes con 2 parches o más:
#      qué fracción de la variación de la varianza del ruido (7x7) se debe a
#      diferencias ENTRE imágenes (1 = todo es la imagen; 0 = nada).
#   c) Spearman entre la varianza del ruido y la media del parche: si el
#      ruido depende del brillo de la zona.
ids_parches = pd.Series([posicion[0] for posicion in posiciones_elegidas])
por_imagen = ids_parches.value_counts().reindex(df["id"], fill_value=0)
print(f"\nImágenes sin ningún parche homogéneo: {(por_imagen == 0).sum()} de {len(df)} "
      f"({numero(100 * (por_imagen == 0).mean(), 1)} %); con uno solo: {(por_imagen == 1).sum()}; "
      f"con 2 o más: {(por_imagen >= 2).sum()}")

tabla_ruido = pd.DataFrame({"id": ids_parches, "varianza": varianzas_7, "media": medias})
varias = tabla_ruido[tabla_ruido["id"].map(por_imagen) >= 2]
grupos = varias.groupby("id")["varianza"]
k, N = grupos.ngroups, len(varias)
n0 = (N - (grupos.size() ** 2).sum() / N) / (k - 1)                       # parches por imagen (promedio ajustado)
entre = (grupos.size() * (grupos.mean() - varias["varianza"].mean()) ** 2).sum() / (k - 1)
dentro = ((varias["varianza"] - grupos.transform("mean")) ** 2).sum() / (N - k)
icc = (entre - dentro) / (entre + (n0 - 1) * dentro)
print(f"Correlación intraclase ICC(1) de la varianza del ruido ({k} imágenes, {N} parches): {numero(icc, 2)}")

rho, p = spearmanr(tabla_ruido["varianza"], tabla_ruido["media"])
texto_p = "p < 0,001" if p < 0.001 else f"p = {numero(p, 3)}"
print(f"Spearman entre la varianza del ruido y la media del parche: {numero(rho, 2)} "
      f"(n = {len(tabla_ruido)}, {texto_p})")
print("  -> Por eso se usa un único valor para todas las imágenes (ver informe).")

fig, eje = plt.subplots(figsize=(8, 4.5))
bordes = np.linspace(0, np.percentile(varianzas_32, 98), 60)
eje.hist(varianzas_32, bins=bordes, color="#8c8c8c", alpha=0.7, label="parche entero 32x32")
eje.hist(varianzas_7, bins=bordes, color="#2a78b5", alpha=0.7, label=f"ventanas {VENTANA}x{VENTANA} (mediana por parche)")
eje.axvline(varianza_ruido_32, color="#555555", linestyle="--", label=f"mediana 32x32 = {numero(varianza_ruido_32, 1)}")
eje.axvline(varianza_ruido, color="#2a78b5", linestyle="--", label=f"mediana {VENTANA}x{VENTANA} = {numero(varianza_ruido, 1)}")
eje.set_xlabel("Varianza del parche (niveles de gris²)")
eje.set_ylabel("Cantidad de parches")
eje.set_title("El mismo ruido medido con dos ventanas")
eje.legend(fontsize=8)
eje.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
guardar_figura("04_varianza_ruido_por_ventana.png")


# ---------------------------------------------------------------------------
# BLOQUE 5. Varianza local por imagen.
# En vez de parches sueltos, miramos toda la imagen: para cada píxel calculamos
# la varianza en una ventana de 7x7 a su alrededor, y nos quedamos con la
# mediana de cada imagen. Así vemos cuánto cambia el "nivel de variación"
# de una imagen a otra. (Se calculó en la misma pasada del bloque 1.)
#   - Varianza local = media de x² − (media de x)², con las medias en la ventana.
#   - Solo se usan ventanas enteras dentro del abanico de la sonda y sin marcas.
# Además: en los píxeles donde la varianza local no supera la varianza del
# ruido, Lee los trata como "solo ruido" y los reemplaza por el promedio
# (peso k = 0). Se informa qué % de cada imagen queda así.
# Figuras: 05 (histograma por imagen) y 05b (mapa en la imagen de ejemplo).
# ---------------------------------------------------------------------------
titulo(f"BLOQUE 5. Varianza local (ventanas de {VENTANA}x{VENTANA}) por imagen")

medianas_varianza = np.array(medianas_varianza)
print(f"Mediana de la varianza local en cada una de las {len(medianas_varianza)} imágenes:")
print(f"  Mínimo:       {numero(medianas_varianza.min(), 1)}")
print(f"  Percentil 25: {numero(np.percentile(medianas_varianza, 25), 1)}")
print(f"  Percentil 50: {numero(np.percentile(medianas_varianza, 50), 1)}")
print(f"  Percentil 75: {numero(np.percentile(medianas_varianza, 75), 1)}")
print(f"  Máximo:       {numero(medianas_varianza.max(), 1)}")

pct_solo_ruido = np.array([100 * (m <= varianza_ruido).mean() for m in muestras_varianza])
pct_solo_ruido_32 = np.array([100 * (m <= varianza_ruido_32).mean() for m in muestras_varianza])
print(f"\n% de cada imagen con varianza local <= varianza del ruido (Lee la promedia entera):")
print(f"  con la referencia de {VENTANA}x{VENTANA} ({numero(varianza_ruido, 1)}): mediana {numero(np.median(pct_solo_ruido), 1)} %")
print(f"  con la de 32x32 ({numero(varianza_ruido_32, 1)}):  mediana {numero(np.median(pct_solo_ruido_32), 1)} %")

fig, eje = plt.subplots(figsize=(7, 4.5))
eje.hist(medianas_varianza, bins=50, color="tab:blue")
eje.axvline(varianza_ruido, color="tab:red", linestyle="--", label=f"varianza del ruido ({VENTANA}x{VENTANA}) = {numero(varianza_ruido, 1)}")
eje.set_xlabel(f"Mediana de la varianza local {VENTANA}x{VENTANA} de la imagen")
eje.set_ylabel("Cantidad de imágenes")
eje.legend()
plt.tight_layout()
guardar_figura("05_varianza_local.png")

# 05b: mapa de varianza local de la imagen de ejemplo
if ejemplo is not None:
    limpia_ej, lesion_ej, _, var_ej, valido_ej = ejemplo
    fig, ejes = plt.subplots(1, 3, figsize=(18, 5.5))
    ejes[0].imshow(limpia_ej, cmap="gray", vmin=0, vmax=255)
    ejes[0].set_title(f"{ID_EJEMPLO} (limpia)")
    mapa = ejes[1].imshow(np.log10(var_ej + 1), cmap="magma")
    plt.colorbar(mapa, ax=ejes[1], fraction=0.035, label="log10(varianza local + 1)")
    ejes[1].set_title(f"Varianza local {VENTANA}x{VENTANA} (claro = mucha variación: bordes)")
    solo_ruido = (var_ej <= varianza_ruido) & valido_ej
    ejes[2].imshow(limpia_ej, cmap="gray", vmin=0, vmax=255)
    capa = np.zeros(limpia_ej.shape + (4,))
    capa[solo_ruido] = [0, 0.8, 1, 0.45]
    ejes[2].imshow(capa)
    ejes[2].set_title(f"Celeste: varianza <= ruido ({numero(100 * solo_ruido.sum() / valido_ej.sum(), 0)} %), "
                      f"Lee la promedia entera")
    for eje in ejes:
        if lesion_ej.any():
            eje.contour(lesion_ej, levels=[0.5], colors="yellow", linewidths=1)
        eje.axis("off")
    plt.tight_layout()
    guardar_figura("05b_mapa_varianza_local.png", dpi=100)