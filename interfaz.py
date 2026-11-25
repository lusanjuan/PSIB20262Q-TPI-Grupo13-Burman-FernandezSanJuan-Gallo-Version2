"""Interfaz gráfica (objetivo opcional del anteproyecto).

Permite cargar una ecografía, marcar la lesión con un clic (la semilla de la segmentación) y ver
cada paso del método: las marcas del ecógrafo que se borraron (paso 2), el filtro de Lee
(pasos 3 y 4), la segmentación (paso 5) y los descriptores con la clasificación
benigna/maligna (paso 6).

Uso, desde la carpeta del proyecto y con el entorno activado:

    streamlit run interfaz.py

Necesita haber corrido antes los pasos 1 a 6 (usa sus resultados guardados en outputs/).
Se abre en el navegador. No reemplaza a los scripts: es una forma de ver el método funcionando
sobre una imagen por vez. El método en sí está en metodo.py.
"""

import hashlib
import io
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colormaps
from PIL import Image, ImageDraw
from scipy import ndimage
from streamlit_image_coordinates import streamlit_image_coordinates

import comun
import metodo
from comun import armar_tabla, agregar_grupos, cargar
from filtros import USAR_RUIDO_PASO3

ANCHO_CLIC = 620                 # px de pantalla de la imagen donde se hace clic
COLOR_SEGMENTACION = (0, 230, 0)
COLOR_MANUAL = (255, 215, 0)
COLOR_MARCAS = (255, 60, 30)
LADO_ENTRENAMIENTO = metodo.LADO  # la ventana elegida en el paso 5, con la que se entrenó el clasificador
TOLERANCIA_ENTRENAMIENTO = int(metodo.TOLERANCIA)   # la tolerancia elegida en el paso 5

NOMBRES_DESCRIPTORES = {
    "circularidad": "Circularidad", "alto/ancho": "Alto / ancho", "ecogenicidad": "Ecogenicidad",
    "sombra": "Sombra posterior", "homogeneidad": "Homogeneidad",
}


# ---- Carga (con caché: se calcula una sola vez) -------------------------------

def version_del_metodo():
    """Huella de los archivos de los que depende el resultado. Se pasa a las
    funciones con caché: si alguno cambia (por ejemplo, después de volver a correr
    los pasos 5 y 6 y subirlos), la caché se descarta y se recalcula todo, en vez
    de seguir mostrando el modelo viejo."""
    huella = hashlib.md5()
    for ruta in [metodo.RUTA_DESCRIPTORES, metodo.RUTA_PARAMETROS, metodo.RUTA_GRUPOS, metodo.__file__]:
        if os.path.exists(ruta):
            with open(ruta, "rb") as archivo:
                huella.update(archivo.read())
    return huella.hexdigest()


@st.cache_resource(show_spinner="Entrenando el clasificador con los descriptores del paso 6...")
def entrenar_modelo(version):
    return metodo.entrenar()


def cargar_modelo():
    return entrenar_modelo(version_del_metodo())


CARPETA_EJEMPLOS = "ejemplos"


@st.cache_data
def casos_busi():
    """Casos con lesión de BUSI (data/raw), sin las imágenes con etiqueta contradictoria
    del paso 1. Si el dataset no está (por ejemplo, en la versión publicada en internet),
    usa la carpeta ejemplos/ si existe, con la misma estructura (ejemplos/benign/,
    ejemplos/malignant/, con sus máscaras). Si tampoco está, devuelve None y solo se
    puede subir una imagen. Devuelve (casos, son_ejemplos)."""
    son_ejemplos = False
    try:
        casos = armar_tabla()
    except Exception:
        casos = pd.DataFrame()
    if casos.empty and os.path.isdir(CARPETA_EJEMPLOS):
        carpeta_original = comun.CARPETA_DATOS
        comun.CARPETA_DATOS = CARPETA_EJEMPLOS
        try:
            casos, son_ejemplos = armar_tabla(), True
        except Exception:
            casos = pd.DataFrame()
        finally:
            comun.CARPETA_DATOS = carpeta_original
    if casos.empty:
        return None, False
    if not son_ejemplos:
        try:
            casos = agregar_grupos(casos)
        except FileNotFoundError:
            pass   # sin el paso 1 se muestran todas
    casos = casos[casos["clase"] != "normal"]
    return (None, False) if casos.empty else (casos.set_index("id"), son_ejemplos)


@st.cache_data(show_spinner="Limpiando, filtrando, segmentando y calculando descriptores...")
def analizar_con_cache(img, semilla, lado, tolerancia, version):
    return metodo.analizar(img, semilla, lado, cargar_modelo(), tolerancia)


def analizar(img, semilla, lado, tolerancia):
    """Corre el método completo sobre una imagen, una semilla (fila, columna), una ventana
    de `lado` px y una tolerancia del region growing."""
    return analizar_con_cache(img, semilla, lado, tolerancia, version_del_metodo())


def toca_ventana(mascara, ventana):
    """True si la segmentación llega a un borde de la ventana que no es borde de la imagen (quedó cortada)."""
    filas, cols = ventana
    h, w = mascara.shape
    return bool((filas.start > 0 and mascara[filas.start, cols].any())
                or (filas.stop < h and mascara[filas.stop - 1, cols].any())
                or (cols.start > 0 and mascara[filas, cols.start].any())
                or (cols.stop < w and mascara[filas, cols.stop - 1].any()))


# ---- Dibujo -------------------------------------------------------------------

def a_rgb(img):
    return np.dstack([np.clip(img, 0, 255).astype(np.uint8)] * 3)


def pintar(rgb, mascara, color, alfa=1.0):
    """Pinta los píxeles de `mascara` de `color` (alfa = opacidad)."""
    rgb = rgb.astype(float)
    rgb[mascara] = (1 - alfa) * rgb[mascara] + alfa * np.array(color)
    return rgb.astype(np.uint8)


def contorno(mascara, grosor=2):
    """Borde de la máscara, de `grosor` px, para dibujarlo sobre la imagen."""
    return mascara & ~ndimage.binary_erosion(mascara, iterations=grosor)


def con_semilla(rgb, semilla, radio=7):
    imagen = Image.fromarray(rgb)
    dibujo = ImageDraw.Draw(imagen)
    y, x = semilla
    dibujo.line([(x - radio, y), (x + radio, y)], fill=(255, 0, 0), width=3)
    dibujo.line([(x, y - radio), (x, y + radio)], fill=(255, 0, 0), width=3)
    return imagen


def mapa_en_color(mapa, mascara=None, minimo=None, maximo=None):
    """Mapa con la escala viridis (por defecto, del percentil 1 al 99) y el contorno de la lesión."""
    if minimo is None or maximo is None:
        minimo, maximo = np.percentile(mapa, [1, 99])
    normalizado = np.clip((mapa - minimo) / max(maximo - minimo, 1e-12), 0, 1)
    rgb = (colormaps["viridis"](normalizado)[..., :3] * 255).astype(np.uint8)
    if mascara is not None and mascara.any():
        rgb = pintar(rgb, contorno(mascara), (255, 40, 40))
    return rgb


def recorte(mascara, forma, margen=60):
    filas, cols = np.nonzero(mascara)
    return (slice(max(filas.min() - margen, 0), min(filas.max() + margen + 1, forma[0])),
            slice(max(cols.min() - margen, 0), min(cols.max() + margen + 1, forma[1])))


# ---- Página -------------------------------------------------------------------

st.set_page_config(page_title="Ecografía de mama – Grupo 13", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.pasos {display: flex; gap: .6rem; flex-wrap: wrap; margin: .2rem 0 1.2rem 0;}
.paso {background: #E6F2F1; color: #0F5E57; border-radius: 999px; padding: .3rem .9rem; font-size: .92rem;}
.tarjeta {border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: .8rem;}
.tarjeta h2 {margin: 0; font-size: 2.1rem; letter-spacing: .04em;}
.tarjeta p {margin: .2rem 0 0 0;}
.benigna {background: #E7F6EC; border: 1px solid #9ED9B0; color: #14532D;}
.maligna {background: #FDECEC; border: 1px solid #F3A6A6; color: #7F1D1D;}
.barra {position: relative; height: 12px; border-radius: 6px; margin: .7rem 0 .2rem 0;
        background: linear-gradient(90deg, #4CAF50 0%, #F2C94C 50%, #E5484D 100%);}
.marca {position: absolute; top: -5px; width: 4px; height: 22px; background: #1E293B; border-radius: 2px;}
.umbral {position: absolute; top: -3px; width: 2px; height: 18px; background: #64748B;}
.escala {display: flex; justify-content: space-between; font-size: .78rem; opacity: .75;}
.dato {background: #F1F5F9; color: #0F172A; border-radius: 12px; padding: .8rem 1rem; height: 100%;}
.dato b {font-size: 1.5rem;}
.leyenda span {display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin: 0 .3rem 0 .8rem; vertical-align: -1px;}
</style>
""", unsafe_allow_html=True)

st.title("Lesiones en ecografía de mama")
st.markdown("Segmentación y clasificación **benigna / maligna** a partir de un clic · PSIB 2026Q2 – TPI – Grupo 13")
st.markdown('<div class="pasos"><div class="paso">1. Elegí una ecografía (barra de la izquierda)</div>'
            '<div class="paso">2. Hacé clic dentro de la lesión</div>'
            '<div class="paso">3. Mirá el resultado y cada paso del método</div></div>', unsafe_allow_html=True)

try:
    entrenado = cargar_modelo()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()
casos, son_ejemplos = casos_busi()
m = entrenado["metricas"]

# 1. Elegir la imagen
with st.sidebar:
    st.header("1. Ecografía")
    opciones = ["Subir una imagen"] if casos is None else ["Del dataset BUSI", "Subir una imagen"]
    origen = st.radio("Origen", opciones, horizontal=True, label_visibility="collapsed")
    mascara_manual, clase_real = None, None
    if origen == "Del dataset BUSI":
        clase = st.segmented_control("Tipo de lesión (según el dataset)", ["Benignas", "Malignas"],
                                     default="Benignas", key="clase_elegida") or "Benignas"
        codigo = "benign" if clase == "Benignas" else "malignant"
        ids = sorted(casos.index[casos["clase"] == codigo], key=lambda s: int(s.split("(")[1].rstrip(")")))
        if st.session_state.get("id_caso") not in ids:
            st.session_state["id_caso"] = ids[0]
        if st.button("Otra imagen al azar", width="stretch"):
            st.session_state["id_caso"] = str(np.random.choice(ids))
        id_caso = st.selectbox(f"Imagen ({len(ids)} {clase.lower()})", ids, key="id_caso")
        if son_ejemplos:
            st.caption("Versión en línea: imágenes de ejemplo de BUSI (el dataset completo no está en GitHub). "
                       "También podés subir una imagen propia.")
        fila = casos.loc[id_caso].copy()
        fila["id"] = id_caso
        img, mascara_manual = cargar(fila)
        nombre, clase_real = id_caso, ("benigna" if codigo == "benign" else "maligna")
        mostrar_manual = st.toggle("Mostrar el contorno del radiólogo (amarillo)", value=False)
    else:
        archivo = st.file_uploader("Ecografía (PNG o JPG)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        if archivo is None:
            st.info("Subí una imagen para empezar.")
            st.stop()
        img = np.array(Image.open(io.BytesIO(archivo.getvalue())).convert("L"))
        nombre = archivo.name
        mostrar_manual = False

    st.header("2. Clic en la lesión")
    st.write("Hacé **un clic dentro de la lesión** (la zona oscura) en la imagen grande.")
    if mascara_manual is not None and mascara_manual.any() and st.button(
            "Elegir el punto por mí", width="stretch",
            help="Usa el punto más central del contorno del radiólogo (el mismo clic simulado del paso 5)."):
        st.session_state["semilla"] = (nombre, tuple(int(v) for v in metodo.semilla_desde_mascara(mascara_manual)))

    lado = st.select_slider(
        "Tamaño de la ventana (px)", options=[150, 200, 250, 300, 400, 500], value=LADO_ENTRENAMIENTO,
        help="La segmentación crece solo dentro de un cuadrado de este lado centrado en el clic. Si la lesión es "
             "más grande, queda cortada en línea recta: agrandá la ventana. Con ventanas grandes puede derramarse "
             f"al tejido de alrededor. El modelo se entrenó con {LADO_ENTRENAMIENTO} px.")
    tolerancia = st.select_slider(
        "Tolerancia (niveles de gris)", options=[15, 20, 25, 30, 35], value=TOLERANCIA_ENTRENAMIENTO,
        help="Cuánto puede diferir el gris de un vecino del promedio de la región para sumarse. Más baja: la "
             "región crece menos (sirve si se derrama al tejido de alrededor). Más alta: crece más (sirve si "
             f"queda chica). El modelo se entrenó con {TOLERANCIA_ENTRENAMIENTO}.")
    st.caption("Si el contorno **se derrama** fuera de la lesión: achicá la ventana o bajá la tolerancia. "
               "Si queda **cortado en línea recta**: agrandá la ventana.")

    st.divider()
    st.caption(f"**Qué tan bien anda:** en {m['lesiones']} lesiones de BUSI (validación cruzada agrupada) detecta "
               f"el {100 * m['sensibilidad']:.0f} % de las malignas y da bien el {100 * m['especificidad']:.0f} % "
               f"de las benignas (AUC {m['auc']:.2f}). El resultado cambia según dónde se haga clic, y supone que hay "
               "una lesión. "
               "Es una herramienta de apoyo, **no un diagnóstico**.")

# la semilla guardada solo vale para la imagen en la que se marcó
semilla = None
if st.session_state.get("semilla") and st.session_state["semilla"][0] == nombre:
    semilla = st.session_state["semilla"][1]

col_clic, col_resultado = st.columns([1.15, 1], gap="large")

with col_clic:
    st.subheader(nombre)
    base = a_rgb(img)
    if mostrar_manual and mascara_manual is not None:
        base = pintar(base, contorno(mascara_manual), COLOR_MANUAL)
    imagen_clic = con_semilla(base, semilla) if semilla else Image.fromarray(base)
    ancho = min(ANCHO_CLIC, img.shape[1])
    clic = streamlit_image_coordinates(imagen_clic, width=ancho, key=f"clic_{nombre}", cursor="crosshair")
    if clic is not None:
        escala = img.shape[1] / clic["width"]
        nueva = (min(int(round(clic["y"] * escala)), img.shape[0] - 1),
                 min(int(round(clic["x"] * escala)), img.shape[1] - 1))
        if nueva != semilla:
            st.session_state["semilla"] = (nombre, nueva)
            st.rerun()
    st.caption("Cruz roja: tu clic. Hacé clic de nuevo para moverlo.")

if semilla is None:
    with col_resultado:
        st.subheader("Resultado")
        st.info("Hacé clic dentro de la lesión (en la imagen de la izquierda) para ver la segmentación y la clasificación.")
    st.stop()

resultado = analizar(img, semilla, lado, tolerancia)
mascara = resultado["mascara"]

with col_resultado:
    st.subheader("Resultado")
    if resultado["clase"] is None:
        st.error(resultado["error"])
        st.stop()
    if toca_ventana(mascara, resultado["ventana"]):
        st.warning(f"La segmentación llegó al borde de la ventana de {lado} px: si ves bordes rectos, la lesión "
                   "quedó cortada. Probá agrandar la ventana en la barra de la izquierda.")
    if lado != LADO_ENTRENAMIENTO or tolerancia != TOLERANCIA_ENTRENAMIENTO:
        st.caption(f"Atención: ventana de {lado} px y tolerancia {tolerancia}. El modelo se entrenó con "
                   f"{LADO_ENTRENAMIENTO} px y tolerancia {TOLERANCIA_ENTRENAMIENTO}, así que la probabilidad "
                   "es menos confiable.")
    p = resultado["probabilidad_maligna"]
    u = entrenado["umbral"]
    estilo = "maligna" if resultado["clase"] == "maligna" else "benigna"
    st.markdown(f"""
<div class="tarjeta {estilo}">
  <p>El modelo dice</p>
  <h2>{resultado['clase'].upper()}</h2>
  <div class="barra"><div class="umbral" style="left: calc({100 * u:.1f}% - 1px);"></div>
  <div class="marca" style="left: calc({100 * p:.1f}% - 2px);"></div></div>
  <div class="escala"><span>benigna</span><span>probabilidad de maligna: <b>{100 * p:.0f} %</b>
  (umbral {100 * u:.0f} %)</span><span>maligna</span></div>
</div>""", unsafe_allow_html=True)

    if clase_real is not None:
        coincidencia = metodo.dice(mascara, metodo.referencia_manual(mascara_manual, semilla))
        acierto = "el modelo acertó" if clase_real == resultado["clase"] else "el modelo se equivocó"
        a, b = st.columns(2)
        a.markdown(f'<div class="dato">Diagnóstico real<br><b>{clase_real}</b><br>{acierto}</div>',
                   unsafe_allow_html=True)
        b.markdown(f'<div class="dato">Coincidencia con el radiólogo<br><b>{100 * coincidencia:.0f} %</b><br>'
                   f'del contorno</div>', unsafe_allow_html=True)
        st.caption("**Coincidencia con el radiólogo** (coeficiente de Dice): cuánto se superpone nuestra segmentación "
                   "(verde) con el contorno que dibujó el médico (amarillo). 100 % = idénticas, 0 % = no se tocan. "
                   "En promedio el método logra alrededor de 65 %.")

    zona = recorte(mascara, img.shape)
    vista = a_rgb(resultado["imagen_limpia"])
    if mostrar_manual and mascara_manual is not None:
        vista = pintar(vista, contorno(mascara_manual), COLOR_MANUAL)
    vista = pintar(vista, contorno(mascara), COLOR_SEGMENTACION)
    st.image(con_semilla(vista, semilla).crop((zona[1].start, zona[0].start, zona[1].stop, zona[0].stop)),
             width="stretch")
    leyenda = '<span style="background:#00E600"></span>segmentación del método'
    if mostrar_manual:
        leyenda += '<span style="background:#FFD700"></span>contorno del radiólogo'
    st.markdown(f'<div class="leyenda" style="font-size:.85rem">{leyenda}'
                f'<span style="background:#FF0000"></span>tu clic</div>', unsafe_allow_html=True)

st.divider()
st.subheader("Cada paso del método")
tab_marcas, tab_lee, tab_desc, tab_como = st.tabs(
    ["1. Marcas del ecógrafo", "2. Filtro de Lee", "3. Descriptores", "Cómo funciona"])

with tab_marcas:
    marcas = resultado["marcas"]
    a, b = st.columns(2)
    a.image(pintar(a_rgb(img), ndimage.binary_dilation(marcas), COLOR_MARCAS),
            caption=f"Original: marcas detectadas en rojo ({100 * marcas.mean():.2f} % de la imagen)",
            width="stretch")
    b.image(resultado["imagen_limpia"], caption="Limpia: las marcas se rellenaron por inpainting biharmónico",
            width="stretch", clamp=True)
    st.caption("Paso 2: top-hat (disco de radio 3) + umbral 100 + descarte de fibras + dilatación de 2 px. "
               "Las marcas sobre tejido claro (más de ~120 de gris) pueden no detectarse.")

with tab_lee:
    columnas = st.columns(3)
    columnas[0].image(mapa_en_color(resultado["imagen_limpia"][zona], mascara[zona], 0, 255),
                      caption="Imagen limpia, sin filtrar", width="stretch")
    columnas[1].image(mapa_en_color(resultado["con_lee"][zona], mascara[zona], 0, 255),
                      caption="Con el filtro de Lee (7×7): sobre esta crece la segmentación", width="stretch")
    columnas[2].image(mapa_en_color(resultado["peso_k"][zona], mascara[zona], 0, 1),
                      caption="Peso k de Lee: violeta = se reemplaza por el promedio, amarillo = se conserva",
                      width="stretch")
    origen_ruido = "medida en el paso 3 en tejido homogéneo" if USAR_RUIDO_PASO3 else "calculada en esta imagen"
    st.caption(f"Pasos 3 y 4: Lee compara la varianza de cada ventana de 7×7 con la del ruido "
               f"({resultado['var_ruido']:.1f}, {origen_ruido}). Donde varía como el ruido, suaviza; donde varía "
               "mucho más (los bordes), conserva. Contorno rojo: la segmentación.")

with tab_desc:
    valores, ref, aportes = resultado["descriptores"], entrenado["referencia"], resultado["aportes"]
    filas = []
    for clave, valor in valores.items():
        aporte = aportes[clave]
        if abs(aporte) < 0.1:
            hacia = "casi nada"
        else:
            hacia = "maligna" if aporte > 0 else "benigna"
        filas.append({"Descriptor": NOMBRES_DESCRIPTORES.get(clave, clave), "Esta lesión": valor,
                      "Mediana benignas": ref["benign"][clave], "Mediana malignas": ref["malignant"][clave],
                      "Aporte": aporte, "Empuja hacia": hacia})
    tabla = pd.DataFrame(filas).set_index("Descriptor")
    st.dataframe(tabla.style.format({"Esta lesión": "{:.3f}", "Mediana benignas": "{:.3f}",
                                     "Mediana malignas": "{:.3f}", "Aporte": "{:+.2f}"}))
    puntaje = resultado["constante"] + sum(aportes.values())
    st.caption(f"Cómo decide: puntaje = constante ({resultado['constante']:+.2f}) + suma de los aportes "
               f"({sum(aportes.values()):+.2f}) = **{puntaje:+.2f}**. Probabilidad de maligna = "
               f"1 / (1 + e^(−puntaje)) = **{100 * resultado['probabilidad_maligna']:.0f} %**. "
               "Cada aporte es el peso que el modelo le da al descriptor por cuánto se aleja esta lesión del "
               "promedio (en desvíos): positivo empuja hacia maligna, negativo hacia benigna. Por eso no alcanza "
               "con mirar a qué mediana se parece cada valor: un descriptor puede parecerse a las benignas y "
               "pesar muy poco.")
    st.caption("Paso 6: 5 descriptores de forma (circularidad, alto/ancho), brillo (ecogenicidad), sombra posterior "
               "y textura (homogeneidad), medidos sobre la imagen limpia sin filtrar. Las medianas son de las "
               "lesiones de BUSI segmentadas con este mismo método.")

with tab_como:
    st.markdown(f"""
1. **Limpieza** (paso 2): se detectan las marcas del ecógrafo (*calipers*, textos) con un top-hat y se rellenan
   por *inpainting* biharmónico.
2. **Filtro de Lee** (pasos 3 y 4): suaviza el speckle donde la imagen varía como el ruido y conserva los bordes.
   La varianza del ruido se midió en el paso 3 en parches de tejido homogéneo.
3. **Segmentación** (paso 5): crecimiento de regiones desde el clic sobre la imagen con Lee, tolerancia
   {TOLERANCIA_ENTRENAMIENTO} niveles de gris, dentro de una ventana de {metodo.LADO} px alrededor del clic, y refinamiento
   morfológico. En BUSI da Dice de alrededor de 0,65 en promedio.
4. **Descriptores y clasificación** (paso 6): 5 descriptores y una regresión logística entrenada con las lesiones de
   BUSI segmentadas de la misma forma. Con validación cruzada agrupada: AUC {m['auc']:.2f}, sensibilidad
   {m['sensibilidad']:.2f}, especificidad {m['especificidad']:.2f}.

**Limitaciones:** el resultado cambia según dónde se haga clic; lesiones más grandes que la ventana quedan
cortadas (se puede agrandar, pero el modelo se entrenó con {metodo.LADO} px); los contornos de la segmentación
son más irregulares que los del radiólogo, lo que le quita peso a la circularidad; el modelo supone que hay una
lesión; se entrenó con un solo equipo y un solo hospital. Detalles en los scripts de los pasos 1 a 6.
""")