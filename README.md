# WDSD: segmentación de instancias de tornillos, entregable 1 de Deep Learning

**Autores:** Ivan Berrio, Jose David Castillo y Jaime Besada.

EDA del WEEE Disassembly Screw Dataset (WDSD), planteado como segmentación de instancias: cada tornillo es un objeto que hay que encontrar y delimitar. El EDA está terminado; la revisión de literatura, los tres modelos base y el análisis estadístico están pendientes.

**Empieza por [`notebooks/02_eda_instancias.ipynb`](notebooks/02_eda_instancias.ipynb).** Es el EDA vigente, ya ejecutado, así que se puede leer sin descargar los datos. Su sección 12 resume las decisiones para los modelos y la 13 trae el veredicto.

[`notebooks/01_eda_wdsd.ipynb`](notebooks/01_eda_wdsd.ipynb) es el EDA anterior, orientado a píxeles. Sigue siendo la referencia para lo que no depende del tipo de salida: inventario y huella del dataset, casi duplicados, grupos de escena, partición, intensidad y normalización. Su métrica primaria (Dice de píxel) queda sustituida.

## Estado del entregable

| Sección del enunciado | Peso | Estado |
|---|---|---|
| 1. Problema y dataset | parte del EDA | Hecho, con pendientes abajo |
| 2. EDA | 30 % | Hecho (02 por instancias, 01 para lo general) |
| 3. Literatura y tres modelos base | 30 % | Pendiente |
| 4. Implementación de los modelos | 20 % | Pendiente |
| 5. Benchmark, baseline y estadística | 20 % | Pendiente |

## Lo que debes saber antes de modelar

1. **La verdad de referencia son los polígonos COCO**, rasterizados con `N.etiqueta_coco` (centros de píxel; reproduce exactamente el campo `area`). No uses `ImageDraw.polygon`, que engorda cada tornillo unos 19 px, ni mezcles las máscaras PNG con COCO: son un 16 % más grandes y su IoU con COCO es 0,84.
2. **Métrica primaria: F1 de instancia por imagen con IoU ≥ 0,5**, con `N.metricas_instancia(gt, pred)`. El umbral 0,5 sale del EDA: a 0,75 las dos versiones de la anotación ya discrepan en el 26 % de las imágenes. AP 0,5:0,95 es solo informativa. Declaración completa en `results/eda/declaracion_a_priori_instancias.json`; $\delta = 0{,}03$ es provisional hasta el piloto.
3. **Los modelos base del enunciado sirven tal cual** (FCN, codificador-decodificador, U-Net reducida). Ningún par de tornillos se toca, así que la máscara binaria se convierte en instancias con `N.etiqueta_mascara(prob > umbral, area_min=5)`. Ajusta el umbral en validación.
4. **No uses la partición oficial como principal.** Tiene fuga por escena. Usa `results/eda/particion.csv` (744 / 93 / 108 imágenes, 3.530 / 479 / 389 tornillos). No la regeneres ni consultes test mientras ajustas.
5. **No reduzcas la imagen.** El diámetro mediano de un tornillo es 10,6 px y el 98,8 % son *small* según COCO. Entrena con parches de 512×512 a resolución nativa, con la mitad centrados en tornillos. Evalúa por ventana deslizante de 512 con solape de 64 px y etiqueta componentes sobre la imagen completa, no por ventana.
6. **Da peso a los tornillos pequeños en la pérdida** (Dice + entropía cruzada o focal, con peso por instancia): la mitad más pequeña de los tornillos solo aporta el 17 % de los píxeles.
7. **Normaliza con** `results/eda/normalizacion_train.json`.
8. **Ten cuidado con la inferencia.** Test solo tiene 14 grupos de escena, así que el bootstrap debe remuestrear grupos. Además, test tiene menos tornillos por imagen que train (3,6 frente a 4,7).

## Pendientes

- [ ] Confirmar la licencia con los autores: la página oficial no la declara.
- [ ] Conseguir el DOI del artículo de referencia (Kalitsios et al., ICRCV 2022).
- [ ] Auditar una muestra para estimar tornillos sin anotar.
- [ ] Comprobar qué métricas y particiones usa la literatura sobre WDSD (AP de COCO, partición oficial): podría ser el *gap* a formular.
- [ ] Preparar un entorno con GPU. En el entorno usado para el EDA, TensorFlow 2.10 no detecta la GPU y PyTorch no está instalado.

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

Requiere Python 3.9:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/02_eda_instancias.ipynb
```

El notebook 02 tarda unos 5 minutos la primera vez; el 01, unos 20. Los cálculos por imagen e instancia quedan en caché en `results/eda/` y no se versionan.

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
