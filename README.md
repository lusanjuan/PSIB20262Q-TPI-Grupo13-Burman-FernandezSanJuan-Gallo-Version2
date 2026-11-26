# Segmentación y clasificación de lesiones mamarias en ecografía (BUSI)

TPI de Procesamiento de Señales e Imágenes Biomédicas (ITBA, 2026 Q2), Grupo 13:
Lucía Burman, Lucía Fernandez San Juan y Agustina María Gallo.

A partir de un clic dentro de la lesión, el método borra las marcas del ecógrafo, filtra el
speckle, segmenta la lesión y la clasifica como benigna o maligna, solo con técnicas clásicas.

**Interfaz en línea:** https://psib20262qtpigrupo13.streamlit.app/

## Cómo correrlo

1. Instalar las dependencias (Python 3):

   ```bash
   pip install -r requirements.txt
   ```

2. El dataset BUSI ([Al-Dhabyani et al., 2020](https://doi.org/10.1016/j.dib.2019.104863)) ya está en
   `data/raw/Dataset_BUSI_with_GT/`, con una carpeta por clase.

3. Correr los pasos en orden, desde esta carpeta. Cada uno imprime sus resultados en la terminal y
   guarda las figuras en `outputs/<paso>/`:

   ```bash
   python3 paso1_exploracion.py
   python3 paso2_marcas.py
   python3 paso3_speckle.py
   python3 paso4_filtros.py
   python3 paso5_segmentacion.py
   python3 paso6_clasificador.py
   ```

   Cada paso usa lo que guardó el anterior. Los cálculos más lentos (rellenar las marcas y
   segmentar todas las imágenes) se guardan la primera vez y después se leen; para rehacerlos,
   poner `RECALCULAR = True` al principio del script.

4. Interfaz local (necesita haber corrido los pasos 1 a 6):

   ```bash
   streamlit run interfaz.py
   ```

## Qué hay en cada archivo

| Archivo | Contenido |
| --- | --- |
| `paso1_exploracion.py` | Dataset: clases, tamaños, brillo de la lesión contra su entorno, imágenes repetidas |
| `paso2_marcas.py` | Detección de las marcas del ecógrafo (top-hat) y relleno (inpainting), validado con 600 cruces sintéticas |
| `paso3_speckle.py` | Caracterización del speckle en parches de tejido homogéneo y nivel de ruido para el filtro |
| `paso4_filtros.py` | Filtro de Lee contra la imagen sin filtrar: ruido, borde y textura |
| `paso5_segmentacion.py` | Segmentación desde un clic (crecimiento de regiones, Otsu, k-means) y evaluación con Dice |
| `paso6_clasificador.py` | 5 descriptores, regresión logística y sesgo de las máscaras manuales |
| `marcas.py`, `filtros.py`, `segmentacion.py`, `descriptores.py` | Las funciones que usan los pasos |
| `comun.py` | Lectura del dataset y de las imágenes limpias |
| `metodo.py` | El método completo sobre una imagen (lo usa la interfaz) |
| `interfaz.py` | Interfaz gráfica en Streamlit |

## Resultados principales

| | |
| --- | --- |
| Segmentación (crecimiento de regiones, 441 imágenes de evaluación) | Dice medio 0,66 |
| Clasificación con nuestra segmentación (validación cruzada) | AUC 0,82 |
| Clasificación con la máscara manual | AUC 0,97 |

La diferencia entre las dos últimas filas viene de las máscaras manuales: las benignas están
dibujadas casi como elipses, así que su forma ya trae información del diagnóstico. El detalle está
en el informe.
