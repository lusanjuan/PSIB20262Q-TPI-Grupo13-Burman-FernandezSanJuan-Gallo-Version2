# metodo.py
# El método completo aplicado a UNA imagen, para la interfaz (interfaz.py).
# Usa exactamente las mismas funciones que los pasos 2 a 6:
#   paso 2: detectar_marcas y rellenar (marcas.py)
#   pasos 3 y 4: filtro de Lee con la varianza del ruido del paso 3 (filtros.py)
#   paso 5: region growing desde el clic (segmentacion.py), con los parámetros elegidos
#   paso 6: los 5 descriptores (descriptores.py) y la regresión logística
# El clasificador se entrena con los descriptores que guardó el paso 6
# (columnas "B": calculados con nuestra segmentación, como en un uso real).

import os

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from marcas import detectar_marcas, rellenar
from filtros import filtro_lee, ruido_para_lee, peso_lee
from segmentacion import segmentar, ventana_alrededor
from descriptores import descriptores, mancha_del_clic

# Método y parámetros de la segmentación: los que eligió el paso 5 (los lee de
# outputs/paso5_segmentacion/parametros_elegidos.csv, así siempre coinciden con
# las segmentaciones con las que el paso 6 entrenó el clasificador).
RUTA_PARAMETROS = "outputs/paso5_segmentacion/parametros_elegidos.csv"
if not os.path.exists(RUTA_PARAMETROS):
    raise FileNotFoundError(f"No existe {RUTA_PARAMETROS}. En tu compu: corré el paso 5. En la versión "
                            "publicada: subí ese archivo a GitHub con git add -f.")
_parametros = pd.read_csv(RUTA_PARAMETROS).iloc[0]
METODO = str(_parametros["metodo"])
LADO = int(_parametros["ventana"])
TOLERANCIA = float(_parametros["tolerancia"])

# Umbral de la interfaz: probabilidad >= 0,5 -> maligna. Como la regresión
# logística se entrena con class_weight="balanced" (benignas y malignas pesan lo
# mismo), 0,5 es el punto de equilibrio entre sensibilidad y especificidad.
# (El umbral de sensibilidad >= 0,9 del paso 6 marca como malignas a más de la
# mitad de las imágenes: sirve para el análisis, no para mostrar un resultado.)
UMBRAL = 0.5

DESCRIPTORES = ["circularidad", "alto/ancho", "ecogenicidad", "sombra", "homogeneidad"]
RUTA_DESCRIPTORES = "outputs/paso6_clasificador/descriptores.csv"
RUTA_GRUPOS = "outputs/paso1_exploracion/grupos_repetidas.csv"
SEMILLA = 13


def clasificador():
    """La misma regresión logística del paso 6."""
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))


def entrenar():
    """Entrena el clasificador con los descriptores B del paso 6 y estima cómo anda.
    - Validación cruzada agrupada de 5 pliegues (grupos de repetidas del paso 1):
      cada imagen recibe la probabilidad de un modelo que no la vio. Con esas
      probabilidades se calcula el AUC, y la sensibilidad y la especificidad con
      el umbral UMBRAL.
    - El modelo final se entrena con todas las imágenes.
    Devuelve un diccionario con el modelo, el umbral, las métricas y las
    medianas de cada descriptor por clase (para comparar en la interfaz)."""
    for ruta in [RUTA_DESCRIPTORES, RUTA_GRUPOS]:
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No existe {ruta}. En tu compu: corré los pasos 1 a 6. En la versión "
                                    "publicada: subí ese archivo a GitHub con git add -f (ver requirements.txt).")
    tabla = pd.read_csv(RUTA_DESCRIPTORES)
    columnas = ["B " + d for d in DESCRIPTORES]
    tabla = tabla.dropna(subset=columnas).reset_index(drop=True)
    grupos = pd.read_csv(RUTA_GRUPOS).set_index("id").loc[tabla["id"], "grupo_repetidas"].to_numpy()
    X = tabla[columnas].to_numpy()
    y = (tabla["clase"] == "malignant").astype(int).to_numpy()

    probabilidad = np.zeros(len(y))
    validacion = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    for i_ent, i_pru in validacion.split(X, y, groups=grupos):
        probabilidad[i_pru] = clasificador().fit(X[i_ent], y[i_ent]).predict_proba(X[i_pru])[:, 1]
    umbral = UMBRAL
    prediccion = probabilidad >= umbral

    referencia = {clase: {d: tabla.loc[tabla["clase"] == clase, "B " + d].median() for d in DESCRIPTORES}
                  for clase in ["benign", "malignant"]}
    return {
        "modelo": clasificador().fit(X, y),
        "umbral": float(umbral),
        "metricas": {
            "lesiones": len(y),
            "auc": roc_auc_score(y, probabilidad),
            "sensibilidad": (prediccion & (y == 1)).sum() / (y == 1).sum(),
            "especificidad": (~prediccion & (y == 0)).sum() / (y == 0).sum(),
        },
        "referencia": referencia,
    }


def semilla_desde_mascara(mascara):
    """El punto más adentro de la máscara (el mismo clic simulado del paso 5)."""
    return np.unravel_index(np.argmax(ndimage.distance_transform_edt(mascara)), mascara.shape)


def referencia_manual(mascara, semilla):
    """La lesión manual que contiene el clic; si el clic cae afuera, la más grande."""
    etiquetas, n = ndimage.label(mascara)
    if n == 0:
        return mascara
    if etiquetas[semilla] > 0:
        return etiquetas == etiquetas[semilla]
    tamanios = np.bincount(etiquetas.ravel())[1:]
    return etiquetas == (np.argmax(tamanios) + 1)


def dice(a, b):
    """Dice = 2·|A∩B| / (|A| + |B|). 1 = coinciden exacto, 0 = no se tocan."""
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else 2 * (a & b).sum() / total


def analizar(img, semilla, lado, entrenado, tolerancia=None):
    """Corre el método completo sobre una imagen en gris (0-255), un clic
    (fila, columna), una ventana de `lado` px y una tolerancia del region growing
    (si no se indica, la que eligió el paso 5). Devuelve un diccionario con
    todos los pasos intermedios y el resultado."""
    tolerancia = TOLERANCIA if tolerancia is None else tolerancia
    img = np.asarray(img, dtype=np.uint8)
    semilla = tuple(int(v) for v in semilla)

    # Paso 2: marcas
    marcas = detectar_marcas(img)
    limpia = rellenar(img, marcas)

    # Pasos 3 y 4: Lee
    var_ruido = ruido_para_lee(limpia, marcas)
    con_lee = filtro_lee(limpia, var_ruido)

    # Paso 5: segmentación dentro de la ventana centrada en el clic
    ventana = ventana_alrededor(img.shape, semilla, lado)
    clic_ventana = (semilla[0] - ventana[0].start, semilla[1] - ventana[1].start)
    completa = np.zeros(img.shape, dtype=bool)
    completa[ventana] = segmentar(METODO, con_lee[ventana], clic_ventana, tolerancia)
    mascara = mancha_del_clic(completa, semilla)

    resultado = {
        "marcas": marcas, "imagen_limpia": limpia, "con_lee": con_lee,
        "peso_k": peso_lee(limpia, var_ruido), "var_ruido": var_ruido,
        "mascara": mascara, "ventana": ventana,
        "clase": None, "probabilidad_maligna": None, "descriptores": None, "error": None,
        "aportes": None, "constante": None, "tolerancia": tolerancia,
    }
    if not mascara.any():
        resultado["error"] = "No se pudo segmentar desde ese punto. Probá con un clic más adentro de la lesión."
        return resultado

    # Paso 6: descriptores y clasificación (sobre la imagen limpia sin filtrar, como en el paso 6)
    valores = descriptores(limpia, mascara)
    if any(not np.isfinite(valores[d]) for d in DESCRIPTORES):
        resultado["error"] = ("No se pudo calcular algún descriptor (por ejemplo, la sombra posterior cae "
                              "fuera de la imagen). Probá con otra imagen o con otro clic.")
        return resultado
    x = np.array([[valores[d] for d in DESCRIPTORES]])
    probabilidad = float(entrenado["modelo"].predict_proba(x)[0, 1])
    resultado["descriptores"] = {d: float(valores[d]) for d in DESCRIPTORES}
    # Aporte de cada descriptor al puntaje de la regresión logística:
    #   z = (valor − media) / desvío   (la estandarización del modelo)
    #   aporte = coeficiente · z       (+ empuja hacia maligna, − hacia benigna)
    #   puntaje = constante + suma de aportes;  probabilidad = 1 / (1 + e^(−puntaje))
    escalador, regresion = entrenado["modelo"][0], entrenado["modelo"][-1]
    z = (x[0] - escalador.mean_) / escalador.scale_
    resultado["aportes"] = {d: float(c * zi) for d, c, zi in zip(DESCRIPTORES, regresion.coef_[0], z)}
    resultado["constante"] = float(regresion.intercept_[0])
    resultado["probabilidad_maligna"] = probabilidad
    resultado["clase"] = "maligna" if probabilidad >= entrenado["umbral"] else "benigna"
    return resultado