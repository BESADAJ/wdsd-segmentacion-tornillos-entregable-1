# WDSD: segmentación de instancias de tornillos, entregable 1 de Deep Learning

**Autores:** Ivan Berrio, Jose David Castillo y Jaime Besada.

Segmentación de instancias de tornillos en el WEEE Disassembly Screw Dataset (WDSD): cada tornillo es un objeto que hay que encontrar y delimitar. El entregable 1 está completo: EDA, revisión de literatura, tres modelos base (Mask R-CNN, YOLOv8-seg y SOLOv2, con 5 semillas cada uno) y línea base estadística.

**Empieza por [`notebooks/entrega_E1_instancias.ipynb`](notebooks/entrega_E1_instancias.ipynb).** Es el informe completo en un solo notebook, ya ejecutado, así que se puede leer sin descargar los datos:

1. Introducción.
2. EDA orientado a instancias (el contenido de [`02_eda_instancias.ipynb`](notebooks/02_eda_instancias.ipynb)).
3. Revisión de literatura con PRISMA 2020.
4. Implementación de los modelos base.
5. Línea base estadística.
6. Conclusiones.

Las secciones 3 a 5 también están por separado en [`notebooks/03_modelos_base_instancias.ipynb`](notebooks/03_modelos_base_instancias.ipynb).

El mismo informe se publica como **Jupyter Book** (`_config.yml` y `_toc.yml` en la raíz), con el EDA general [`01_eda_wdsd.ipynb`](notebooks/01_eda_wdsd.ipynb) como anexo. Para construirlo:

```bash
pip install "jupyter-book<2"
jupyter-book build .
```

El libro queda en `_build/html/index.html`. Usa las salidas guardadas en los notebooks, así que no hace falta el dataset ni una GPU.

[`notebooks/01_eda_wdsd.ipynb`](notebooks/01_eda_wdsd.ipynb) es el EDA anterior, orientado a píxeles. Sigue siendo la referencia para lo que no depende del tipo de salida: inventario y huella del dataset, casi duplicados, grupos de escena, partición, intensidad y normalización. Su métrica primaria (Dice de píxel) queda sustituida.

## Estado del entregable

| Sección del enunciado | Peso | Estado |
|---|---|---|
| 1. Problema y dataset | parte del EDA | Hecho, con pendientes abajo |
| 2. EDA | 30 % | Hecho (02 por instancias, 01 para lo general) |
| 3. Literatura y tres modelos base | 30 % | Hecho: 9 estudios incluidos (`literatura/instancias/`) |
| 4. Implementación de los modelos | 20 % | Hecho: 15 corridas (`results/instancias/corridas/`) |
| 5. Benchmark, baseline y estadística | 20 % | Hecho (`informe/seccion_5_instancias.md`) |

**Línea base (prueba, 5 semillas, F1 de instancia por imagen):**
- YOLOv8-seg: 0,521 ± 0,061.
- Mask R-CNN: 0,447 ± 0,017.
- SOLOv2: 0,297 ± 0,027.

Por grupos de prueba, SOLOv2 queda por debajo de los otros dos (Holm p ≤ 0,0012). La diferencia entre Mask R-CNN y YOLOv8-seg no es concluyente: con 14 grupos, el efecto mínimo detectable (0,10) es mayor que δ = 0,03.

## Modelos base

- [`ENTRENAMIENTO_INSTANCIAS.md`](ENTRENAMIENTO_INSTANCIAS.md) tiene el protocolo, la instalación en Windows (incluida la de SOLOv2) y los comandos.
- `entrenar_instancias.py` entrena Mask R-CNN y YOLOv8-seg (entorno `requirements_wdsd_dl_congelado.txt`). `solov2_instancias.py` entrena SOLOv2 con MMDetection (entorno `requirements_solov2.txt`).
- `estadistica_instancias.py` hace la comparación por grupos: Friedman, Wilcoxon con Holm, bootstrap y efecto mínimo detectable.
- Cada corrida deja en `results/instancias/corridas/<modelo>_s<semilla>/` sus métricas, el F1 por imagen de prueba y las predicciones en formato COCO. Los pesos no se versionan.
- Las semillas 42 a 44 se corrieron en una RTX 3060 y las 45 y 46 en una RTX 3050, con la misma configuración.

## Lo que debes saber antes de modelar

1. **La verdad de referencia son los polígonos COCO**, rasterizados con `N.etiqueta_coco` (centros de píxel; reproduce exactamente el campo `area`). No uses `ImageDraw.polygon`, que engorda cada tornillo unos 19 px, ni mezcles las máscaras PNG con COCO: son un 16 % más grandes y su IoU con COCO es 0,84.
2. **Métrica primaria: F1 de instancia por imagen con IoU ≥ 0,5**, con `N.metricas_instancia(gt, pred)`. El umbral 0,5 sale del EDA: a 0,75 las dos versiones de la anotación ya discrepan en el 26 % de las imágenes. PQ y AP50 son secundarias y AP 0,5:0,95 es solo informativa. Declaración completa en `results/eda/declaracion_a_priori_instancias.json`, con $\delta = 0{,}03$.
3. **Los modelos base del E1 son de instancias** (Mask R-CNN, YOLOv8-seg y SOLOv2). Un modelo semántico (FCN, U-Net) también sirve, y queda como candidato para el E2: ningún par de tornillos se toca, así que la máscara binaria se convierte en instancias con `N.etiqueta_mascara(prob > umbral, area_min=5)`. Ajusta el umbral en validación.
4. **No uses la partición oficial como principal.** Tiene fuga por escena. Usa `results/eda/particion.csv` (744 / 93 / 108 imágenes, 3.530 / 479 / 389 tornillos). No la regeneres ni consultes test mientras ajustas.
5. **No reduzcas la imagen.** El diámetro mediano de un tornillo es 10,6 px y el 98,8 % son *small* según COCO. Los modelos base usan la imagen completa de 1280×720 y anclas pequeñas. Para un modelo semántico, entrena con parches de 512×512 a resolución nativa, con la mitad centrados en tornillos, evalúa por ventana deslizante de 512 con solape de 64 px y etiqueta componentes sobre la imagen completa, no por ventana.
6. **Da peso a los tornillos pequeños en la pérdida** (Dice + entropía cruzada o focal, con peso por instancia): la mitad más pequeña de los tornillos solo aporta el 17 % de los píxeles.
7. **Normaliza con** `results/eda/normalizacion_train.json`.
8. **Ten cuidado con la inferencia.** Test solo tiene 14 grupos de escena, así que el bootstrap debe remuestrear grupos. Además, test tiene menos tornillos por imagen que train (3,6 frente a 4,7).

## Pendientes

- [ ] Confirmar la licencia con los autores: la página oficial no la declara.
- [x] Conseguir el DOI del artículo de referencia (Kalitsios et al., ICRCV 2022): 10.1109/ICRCV55858.2022.9953178.
- [ ] Auditar una muestra para estimar tornillos sin anotar.
- [x] Comprobar qué métricas y particiones usa la literatura: ninguno de los 9 estudios repite corridas ni hace pruebas estadísticas, y solo uno usa partición agrupada (sección 3 del informe).
- [x] Preparar un entorno con GPU: ver `ENTRENAMIENTO_INSTANCIAS.md`.

## Datos

El dataset no está en este repositorio porque pesa 2,7 GB y no tiene licencia declarada.

1. Descarga la carpeta `WDSD` desde la [página oficial](https://vcl.iti.gr/dataset/weee-disassembly-screw-dataset/). Google Drive la entrega en dos archivos zip, y **hacen falta los dos**: cada uno trae cerca de la mitad de las imágenes y de las máscaras, y el COCO de train viene en el segundo.
2. Extrae ambos junto a este repositorio:

```text
carpeta_de_trabajo/
├── WDSD-<fecha>-1-001/WDSD/
├── WDSD-<fecha>-1-002/WDSD/
└── wdsd-segmentacion-tornillos-entregable-1/
```

Debes tener 945 pares y 4.398 instancias COCO. El SHA-256 esperado del dataset, según `E.huella_dataset`, está en `results/eda/declaracion_a_priori.json`.

## Ejecución

Para el EDA, con Python 3.9:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/02_eda_instancias.ipynb
```

El notebook 02 tarda unos 5 minutos la primera vez; el 01, unos 20. Los cálculos por imagen e instancia quedan en caché en `results/eda/` y no se versionan.

Los notebooks 03 y de entrega no entrenan nada: leen las corridas guardadas en `results/instancias/corridas/`. Para reentrenar, sigue `ENTRENAMIENTO_INSTANCIAS.md`.

## Uso de la partición y de la métrica

```python
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from wdsd_e1 import eda as E, instancias as N

particion = pd.read_csv("results/eda/particion.csv")
coco = N.por_archivo(E.carga_coco())        # archivo -> lista de anotaciones

gt = N.etiqueta_coco(coco["frame_653.png"])  # etiquetas 0..K de la verdad
pred = N.etiqueta_mascara(prob > umbral, area_min=5)
N.metricas_instancia(gt, pred, umbral=0.5)  # tp, fp, fn, f1, sq, pq
```

La columna `grupo` identifica la unidad de independencia y `familia` la familia de escena. `src/wdsd_e1/eda.py` reúne el inventario y las medidas del EDA 01; `src/wdsd_e1/instancias.py`, las del 02.
