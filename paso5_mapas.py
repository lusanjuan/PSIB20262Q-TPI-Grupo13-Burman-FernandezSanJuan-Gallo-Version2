# paso5_mapas.py
# Mapas de brillo y de textura: ¿cuáles separan mejor la lesión del tejido
# de alrededor? Elegimos 3 para el paso 6.
# Se usan las 647 imágenes con lesión (benignas y malignas), limpias de marcas.
# Se corre desde la carpeta del proyecto:  python3 paso5_mapas.py
# Imprime los resultados en la terminal y guarda las figuras en outputs/paso5/.

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no abrir ventanas: solo guardar archivos
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import mannwhitneyu, spearmanr

from comun import armar_tabla, cargar
from marcas import limpiar
from filtros import filtro_mediana, abanico
from mapas import calcular_mapas, recorte_lesion, NOMBRES_MAPAS, VENTANAS

CARPETA_SALIDA = "outputs/paso5"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

df = armar_tabla()
df = df[df["clase"] != "normal"].reset_index(drop=True)   # 647 imágenes con lesión

# Grupos de mapas
GRUPO_A = ["intensidad", "media local"]                                  # brillo (sobre la mediana 7x7)
GRUPO_B = [n for n in NOMBRES_MAPAS if n not in GRUPO_A]                 # textura (sin filtrar)

# Resultados guardados: si existen, se leen en vez de recalcular los mapas
RUTA_AUC = os.path.join(CARPETA_SALIDA, "auc_por_imagen.csv")
RUTA_CORRELACION = os.path.join(CARPETA_SALIDA, "correlacion_mapas.csv")


def titulo(texto):
    """Imprime un título visible en la terminal."""
    print()
    print("=" * 70)
    print(texto, flush=True)
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


def preparar(fila):
    """Carga la imagen, la limpia, la filtra con mediana 7x7 y recorta alrededor de la lesión.
    Devuelve: imagen cruda (limpia, sin filtrar), imagen filtrada, lesión y anillo, todas recortadas.
    También devuelve el interior de la lesión (sin la banda de 3 px del borde)."""
    img, lesion = cargar(fila)
    limpia, _ = limpiar(img)
    filtrada = filtro_mediana(limpia)
    en_abanico = abanico(limpia)

    r = recorte_lesion(lesion)
    lesion, cruda, filtrada, en_abanico = lesion[r], limpia[r], filtrada[r], en_abanico[r]

    contorno = lesion & ~ndimage.binary_erosion(lesion)
    banda = ndimage.binary_dilation(contorno, np.ones((3, 3)))
    interior = lesion & ~banda
    distancia = ndimage.distance_transform_edt(~lesion)
    anillo = (distancia >= 5) & (distancia <= 20) & en_abanico
    return cruda, filtrada, lesion, interior, anillo


# ---------------------------------------------------------------------------
# BLOQUE 1. Mapas.
# Un mapa es una imagen donde cada píxel vale una medida calculada en una
# ventana alrededor (el ejemplo con números está en mapas.py).
# Los 10 mapas:
#   - Brillo, sobre la imagen filtrada con mediana 7x7 (la decisión del paso 4):
#     intensidad (la imagen tal cual, sin ventana) y media local.
#   - Textura, sobre la imagen sin filtrar (el paso 4 mostró que filtrar borra
#     textura): varianza local, coeficiente de variación (desvío / media),
#     gradiente de Sobel promediado en la ventana, y 5 propiedades de la GLCM:
#     contraste, homogeneidad, energía, correlación y entropía.
# GLCM: 16 niveles fijos (nivel = gris // 16), distancia 1, ángulos 0°, 45°,
# 90° y 135°, simétrica, propiedad promediada en los 4 ángulos.
# Ventanas: 7, 15 y 31 px (cada mapa salvo la intensidad, con las 3).
# Para que sea rápido: todo se calcula en un recorte alrededor de la lesión
# (caja de la máscara + 40 px) y la GLCM en una grilla cada 4 px, que después
# se agranda al tamaño del recorte.
# El margen de 40 px alcanza: el anillo llega a 20 px del borde y la ventana
# más grande mira 15 px más allá (20 + 15 = 35 < 40).
# Figura: los 10 mapas de una imagen de ejemplo (ventana 15), con el contorno
# de la lesión.
# ---------------------------------------------------------------------------
titulo("BLOQUE 1. Mapas de una imagen de ejemplo (ventana 15)")

ID_EJEMPLO = "benign (209)"
cruda, filtrada, lesion, interior, anillo = preparar(df[df["id"] == ID_EJEMPLO].iloc[0])
ventanas_15 = {nombre: 15 for nombre in NOMBRES_MAPAS}
mapas = calcular_mapas(filtrada, cruda, ventanas_15)

fig, ejes = plt.subplots(2, 5, figsize=(20, 8))
for eje, nombre in zip(ejes.ravel(), NOMBRES_MAPAS):
    ventana = 0 if nombre == "intensidad" else 15
    eje.imshow(mapas[(nombre, ventana)], cmap="viridis")
    eje.contour(lesion, levels=[0.5], colors="red", linewidths=1)
    eje.set_title(nombre)
    eje.axis("off")
plt.suptitle(f"{ID_EJEMPLO}: mapas con ventana de 15 px (rojo = contorno de la lesión)")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "01_mapas_ejemplo.png")
plt.savefig(ruta_figura, dpi=100)
plt.close()
print(f"Figura guardada en {ruta_figura}")


# ---------------------------------------------------------------------------
# BLOQUE 2. AUC por imagen.
# Para cada imagen y cada mapa comparamos los valores del mapa en:
#   - la lesión: el interior de la máscara, sin la banda de 3 px del borde
#     (en el borde las ventanas mezclan lesión y tejido);
#   - el tejido: un anillo de 5 a 20 px por fuera del borde (dentro del abanico).
# AUC = probabilidad de que un píxel de lesión y uno de tejido elegidos al azar
# queden bien ordenados por el mapa (el de lesión con valor mayor). Se calcula
# con la U de Mann-Whitney: AUC = U / (n_lesión · n_tejido).
#   - AUC = 0,5: el mapa no separa nada (es como tirar una moneda);
#   - AUC = 1 (o 0): separa perfecto (todos los de lesión arriba, o todos abajo).
# AUC efectivo = max(AUC, 1 − AUC), porque nos importa que separe, no hacia
# qué lado. Aparte guardamos si la lesión tiene valores más bajos (AUC < 0,5).
# Los resultados se guardan en auc_por_imagen.csv; si el archivo ya existe,
# se lee y no se recalculan los mapas (borrarlo para recalcular).
# ---------------------------------------------------------------------------
titulo("BLOQUE 2. AUC por imagen (647 imágenes; tarda unos 20 minutos)")

if os.path.exists(RUTA_AUC):
    resultados = pd.read_csv(RUTA_AUC)
    print(f"Resultados leídos de {RUTA_AUC} (no se recalculan los mapas)")
else:
    resultados = []
    for numero_imagen, (_, fila) in enumerate(df.iterrows(), start=1):
        cruda, filtrada, lesion, interior, anillo = preparar(fila)
        if interior.sum() == 0 or anillo.sum() == 0:
            continue   # lesión demasiado chica para tener interior
        mapas = calcular_mapas(filtrada, cruda)
        for (nombre, ventana), mapa in mapas.items():
            valores_lesion = mapa[interior]
            valores_tejido = mapa[anillo]
            u, _ = mannwhitneyu(valores_lesion, valores_tejido)
            auc = u / (len(valores_lesion) * len(valores_tejido))
            resultados.append({
                "id": fila["id"], "clase": fila["clase"], "mapa": nombre, "ventana": ventana,
                "auc_efectivo": max(auc, 1 - auc), "lesion_mas_baja": auc < 0.5,
            })
        if numero_imagen % 50 == 0:
            print(f"  {numero_imagen} de {len(df)}", flush=True)
    resultados = pd.DataFrame(resultados)
    resultados.to_csv(RUTA_AUC, index=False)
    print(f"Resultados guardados en {RUTA_AUC}")

print(f"Imágenes analizadas: {resultados['id'].nunique()} de {len(df)}")


# ---------------------------------------------------------------------------
# BLOQUE 3. Tabla: mejor ventana de cada mapa.
# Para cada mapa elegimos la ventana con mayor mediana del AUC efectivo, e
# informamos esa mediana y en qué % de las imágenes la lesión es más baja que
# el tejido (cerca de 0 % o de 100 % = el sentido es consistente entre imágenes).
# ---------------------------------------------------------------------------
titulo("BLOQUE 3. Mejor ventana de cada mapa")

resumen = (resultados.groupby(["mapa", "ventana"])
           .agg(auc_mediana=("auc_efectivo", "median"), pct_mas_baja=("lesion_mas_baja", "mean"))
           .reset_index())
resumen["pct_mas_baja"] *= 100
mejores = resumen.loc[resumen.groupby("mapa")["auc_mediana"].idxmax()]
mejores = mejores.sort_values("auc_mediana", ascending=False).reset_index(drop=True)

for nombre_grupo, grupo in [("Grupo A, brillo (sobre la mediana 7x7)", GRUPO_A),
                             ("Grupo B, textura (sin filtrar)", GRUPO_B)]:
    print(f"\n{nombre_grupo}")
    print(f"  {'mapa':20s} {'ventana':>8s} {'AUC efectivo (mediana)':>24s} {'% lesión más baja':>19s}")
    for _, m in mejores[mejores["mapa"].isin(grupo)].iterrows():
        print(f"  {m['mapa']:20s} {m['ventana']:8d} {numero(m['auc_mediana'], 3):>24s} "
              f"{numero(m['pct_mas_baja'], 1):>17s} %")

# Dos gráficos de barras lado a lado (uno por grupo), con el mismo eje x
fig, ejes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
for eje, (nombre_grupo, grupo, color) in zip(ejes, [("A: brillo (mediana 7x7)", GRUPO_A, "tab:orange"),
                                                    ("B: textura (sin filtrar)", GRUPO_B, "tab:blue")]):
    filas_grupo = mejores[mejores["mapa"].isin(grupo)]
    etiquetas = [f"{m['mapa']} ({m['ventana']})" if m["ventana"] else m["mapa"] for _, m in filas_grupo.iterrows()]
    eje.barh(etiquetas[::-1], filas_grupo["auc_mediana"][::-1], color=color)
    eje.axvline(0.5, color="gray", linestyle="--", label="0,5 = no separa")
    eje.set_xlim(0.4, 1)
    eje.set_xlabel("Mediana del AUC efectivo (647 imágenes)")
    eje.set_title(f"Grupo {nombre_grupo}")
    eje.legend(loc="lower right")
plt.suptitle("Cuánto separa cada mapa la lesión del tejido (mejor ventana entre paréntesis)")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "03_auc_por_mapa.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"Figura guardada en {ruta_figura}")

mejor_ventana = dict(zip(mejores["mapa"], mejores["ventana"]))


# ---------------------------------------------------------------------------
# BLOQUE 4. Benignas vs. malignas.
# ¿Algún mapa separa mejor la lesión del tejido en una clase que en la otra?
# Para cada mapa (en su mejor ventana) comparamos el AUC efectivo por imagen
# entre benignas y malignas con Mann-Whitney.
# ---------------------------------------------------------------------------
titulo("BLOQUE 4. AUC efectivo: benignas vs. malignas")

print("Test: Mann-Whitney U (dos colas), mapa en su mejor ventana")
for nombre in mejores["mapa"]:
    filas_mapa = resultados[(resultados["mapa"] == nombre) & (resultados["ventana"] == mejor_ventana[nombre])]
    benignas = filas_mapa.loc[filas_mapa["clase"] == "benign", "auc_efectivo"]
    malignas = filas_mapa.loc[filas_mapa["clase"] == "malignant", "auc_efectivo"]
    _, p = mannwhitneyu(benignas, malignas)
    print(f"  {nombre:20s} benignas: n = {len(benignas)}, mediana {numero(benignas.median(), 3):6s}   "
          f"malignas: n = {len(malignas)}, mediana {numero(malignas.median(), 3):6s}   {formato_p(p)}")


# ---------------------------------------------------------------------------
# BLOQUE 5. Mapas repetidos.
# Dos mapas pueden separar bien pero medir casi lo mismo (por ejemplo,
# varianza y coeficiente de variación). Para detectarlo, en 100 imágenes al azar
# (semilla 13) calculamos la correlación de Spearman entre los mapas (cada uno
# en su mejor ventana), píxel a píxel en la lesión y el anillo, y tomamos la
# mediana sobre las 100 imágenes. Spearman usa rangos: mide si dos mapas
# ordenan los píxeles igual, aunque la relación no sea una recta.
# La matriz se guarda en correlacion_mapas.csv; si ya existe, se lee.
# Figuras:
#   a) matriz de correlación solo entre los mapas de textura (grupo B);
#   b) correlación de cada mapa de textura con la media local (textura vs. brillo).
# ---------------------------------------------------------------------------
titulo("BLOQUE 5. Correlación entre mapas (100 imágenes al azar)")

orden = list(mejores["mapa"])   # mismo orden que la tabla
if os.path.exists(RUTA_CORRELACION):
    correlacion = pd.read_csv(RUTA_CORRELACION, index_col=0).loc[orden, orden].to_numpy()
    print(f"Matriz leída de {RUTA_CORRELACION} (no se recalculan los mapas)")
else:
    rng = np.random.default_rng(13)
    ids_muestra = rng.choice(resultados["id"].unique(), size=100, replace=False)
    correlaciones = []
    for id_imagen in ids_muestra:
        cruda, filtrada, lesion, interior, anillo = preparar(df[df["id"] == id_imagen].iloc[0])
        mapas = calcular_mapas(filtrada, cruda, mejor_ventana)
        region = interior | anillo
        columnas = np.column_stack([mapas[(nombre, mejor_ventana[nombre])][region] for nombre in orden])
        rho, _ = spearmanr(columnas)
        correlaciones.append(rho)
    correlacion = np.nanmedian(np.array(correlaciones), axis=0)
    pd.DataFrame(correlacion, index=orden, columns=orden).to_csv(RUTA_CORRELACION)
    print(f"Matriz guardada en {RUTA_CORRELACION}")

textura = [n for n in orden if n in GRUPO_B]          # grupo B, en orden de AUC
indice = {n: orden.index(n) for n in orden}

# a) Matriz de correlación entre los mapas de textura
matriz_textura = correlacion[np.ix_([indice[n] for n in textura], [indice[n] for n in textura])]
fig, eje = plt.subplots(figsize=(8.5, 7))
imagen = eje.imshow(matriz_textura, cmap="RdBu_r", vmin=-1, vmax=1)
eje.set_xticks(range(len(textura)))
eje.set_xticklabels(textura, rotation=45, ha="right")
eje.set_yticks(range(len(textura)))
eje.set_yticklabels(textura)
for i in range(len(textura)):
    for j in range(len(textura)):
        eje.text(j, i, numero(matriz_textura[i, j], 2), ha="center", va="center", fontsize=8)
plt.colorbar(imagen, label="ρ de Spearman (mediana de 100 imágenes)")
eje.set_title("Correlación entre mapas de textura (cada uno en su mejor ventana)")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "05a_correlacion_textura.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"Figura guardada en {ruta_figura}")

# b) Correlación de cada mapa de textura con la media local
rho_con_media = [correlacion[indice[n], indice["media local"]] for n in textura]
fig, eje = plt.subplots(figsize=(8, 5))
eje.barh(textura[::-1], rho_con_media[::-1], color="tab:blue")
eje.axvline(0.7, color="gray", linestyle=":", label="±0,7")
eje.axvline(-0.7, color="gray", linestyle=":")
eje.axvline(0, color="black", linewidth=0.8)
eje.set_xlim(-1, 1)
eje.set_xlabel("ρ de Spearman con la media local (mediana de 100 imágenes)")
eje.set_title("Relación entre cada mapa de textura y el brillo")
eje.legend(loc="lower right")
plt.tight_layout()
ruta_figura = os.path.join(CARPETA_SALIDA, "05b_correlacion_con_media_local.png")
plt.savefig(ruta_figura, dpi=120)
plt.close()
print(f"Figura guardada en {ruta_figura}")

print("\nMatriz de correlación entre mapas de textura (ρ de Spearman):")
ancho = 9
print(" " * 20 + "".join(f"{n[:ancho - 1]:>{ancho}s}" for n in textura))
for i, n in enumerate(textura):
    print(f"{n:20s}" + "".join(f"{numero(matriz_textura[i, j], 2):>{ancho}s}" for j in range(len(textura))))

print("\nCorrelación de cada mapa de textura con la media local:")
for n, rho in zip(textura, rho_con_media):
    print(f"  {n:20s} ρ = {numero(rho, 2)}")


# ---------------------------------------------------------------------------
# BLOQUE 6. Elegir 3 mapas para el paso 6.
# Recorremos los mapas de mayor a menor AUC y nos quedamos con uno si:
#   - su AUC efectivo (mediana) es >= 0,70: separa bastante;
#   - la lesión es más baja (o más alta) en al menos el 75 % de las imágenes:
#     el sentido es consistente, así que sirve con una regla fija;
#   - su correlación |ρ| con los ya elegidos es < 0,7: aporta algo nuevo.
# ---------------------------------------------------------------------------
titulo("BLOQUE 6. Mapas elegidos para el paso 6")

elegidos = []
for _, m in mejores.iterrows():
    consistencia = max(m["pct_mas_baja"], 100 - m["pct_mas_baja"])
    i = orden.index(m["mapa"])
    rho_max = max([abs(correlacion[i, orden.index(e)]) for e in elegidos], default=0)

    # Cada condición, con su valor y si se cumple
    print(f"\n  {m['mapa']} (ventana {m['ventana']})")
    print(f"    AUC efectivo = {numero(m['auc_mediana'], 3)} -> "
          f"{'cumple' if m['auc_mediana'] >= 0.70 else 'NO cumple'} (>= 0,70)")
    print(f"    mismo sentido en {numero(consistencia, 1)} % -> "
          f"{'cumple' if consistencia >= 75 else 'NO cumple'} (>= 75 %)")
    if elegidos:
        for e in elegidos:
            rho = correlacion[i, orden.index(e)]
            print(f"    ρ con {e} = {numero(rho, 2)} -> "
                  f"{'cumple' if abs(rho) < 0.7 else 'NO cumple'} (|ρ| < 0,7)")
    else:
        print("    todavía no hay elegidos: no se compara la correlación")

    if m["auc_mediana"] < 0.70:
        motivo = "AUC < 0,70"
    elif consistencia < 75:
        motivo = f"sentido consistente en solo {numero(consistencia, 1)} % de las imágenes"
    elif rho_max >= 0.7:
        motivo = f"|ρ| = {numero(rho_max, 2)} con uno ya elegido"
    else:
        motivo = None
        elegidos.append(m["mapa"])
    estado = "ELEGIDO" if motivo is None else f"descartado ({motivo})"
    print(f"    => {estado}")
    if len(elegidos) == 3:
        break

print("\nMapas elegidos:")
for nombre in elegidos:
    m = mejores[mejores["mapa"] == nombre].iloc[0]
    ventana = "sin ventana" if m["ventana"] == 0 else f"ventana {m['ventana']} px"
    sentido = "más baja" if m["pct_mas_baja"] >= 50 else "más alta"
    print(f"  {nombre} ({ventana}): AUC {numero(m['auc_mediana'], 3)}, "
          f"lesión {sentido} que el tejido en {numero(max(m['pct_mas_baja'], 100 - m['pct_mas_baja']), 1)} % de las imágenes")
if len(elegidos) < 3:
    print(f"  Solo {len(elegidos)} mapas cumplen las 3 condiciones.")
