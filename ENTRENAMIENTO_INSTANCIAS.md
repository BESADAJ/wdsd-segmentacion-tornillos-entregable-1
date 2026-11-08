# Entrenamiento de los modelos base de instancias (Mask R-CNN y YOLOv8-seg)

SOLOv2 va aparte (MMDetection). Ver la sección final.

## Archivos

- `entrenar_instancias.py`: comandos `preparar`, `piloto`, `cola` y `resumen`.
- `src/wdsd_inst/datos.py`: COCO reagrupado con la partición congelada y formato YOLO.
- `src/wdsd_inst/maskrcnn.py`: Mask R-CNN de torchvision.
- `src/wdsd_inst/yolo.py`: YOLOv8-seg de ultralytics.
- `src/wdsd_inst/metricas.py`: AP de máscara (COCOeval) y PQ por imagen. Es la misma evaluación para los tres modelos.

## Pasos en Windows (RTX 3060)

Se corre desde la carpeta que tiene `results/eda/particion.csv` y las carpetas `train/`, `val/`, `test/` y `Annotations/`. La terminal de VS Code no tiene conda en el PATH, así que se usa la ruta completa del Python del entorno. Los comandos son para PowerShell, la terminal por defecto de VS Code.

```powershell
cd "D:\Documents\Personal\Uninorte\Maestría\Primer semestre\Deep Learning\Trabajo\datos\wdsd"
$PY = "C:\Users\ivanj\miniconda3\envs\wdsd_dl\python.exe"

& $PY -m pip install ultralytics pycocotools
# pip puede cambiar torch por la versión de PyPI (solo CPU en Windows). Reponer la de CUDA:
& $PY -m pip install --force-reinstall --no-deps torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
& $PY -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
& $PY entrenar_instancias.py preparar --data-root .
& $PY entrenar_instancias.py piloto --horas 10
```

- La comprobación debe imprimir `2.11.0+cu128 0.26.0+cu128 True`. Los scripts se niegan a entrenar si no ven la GPU (salvo `--permitir-cpu`).
- `preparar` tarda segundos. Debe imprimir 744/93/108 imágenes y 3530/479/389 instancias, con 146/26/14 grupos.
- `piloto` entrena 1 época de cada modelo con la semilla 42 y descarga los pesos COCO (requiere internet). Al final imprime una tabla con segundos por época, memoria pico y cuántas épocas caben en `--horas`.
- Si Mask R-CNN se queda sin memoria (`CUDA out of memory`), repetir con `--lote-maskrcnn 1`. La tasa de aprendizaje se ajusta sola (0.0025 × lote).

## Cola de 5 semillas × 2 modelos

```powershell
& $PY entrenar_instancias.py cola --hasta 07:30 --epocas-maskrcnn 12 --epocas-yolo 40
```

- Las épocas se fijan con lo que diga el piloto.
- Para repartir entre máquinas se usan semillas, no modelos: por ejemplo, `--semillas 42,43,44` en la 3060 y `--semillas 45,46` en la otra máquina. Así el efecto de la máquina cae igual sobre ambos modelos.
- Si se corta, al relanzar la cola salta las corridas que ya terminaron.
- Al final, `& $PY entrenar_instancias.py resumen` escribe `results/instancias/resumen.csv`.

## Cola completa de la terna (recomendada)

`cola_terna.ps1` corre por rondas de semilla con los 3 modelos (semilla 42 de Mask R-CNN, YOLOv8-seg y SOLOv2, luego 43, ...). Si el tiempo se acaba, todos los modelos quedan con el mismo número de semillas. Requiere los dos entornos (`wdsd_dl` y `wdsd_mmdet`).

```powershell
powershell -ExecutionPolicy Bypass -File .\cola_terna.ps1 -Hasta 18:00 -EpMaskRcnn 12 -EpYolo 40 -EpSolo 12
```

- Las épocas se fijan con los pilotos de los tres modelos.
- Los lotes van con `-LoteMaskRcnn 2 -LoteYolo 4 -LoteSolo 2` (valores por defecto) y deben ser iguales en todas las máquinas. Si un modelo no cabe en la 3050 (6 GB), se baja su lote en las dos.
- Para repartir entre máquinas, cada una corre semillas distintas con los 3 modelos y la misma configuración: por ejemplo, `-Semillas 42,43,44` en la 3060 y `-Semillas 45,46` en la otra.
- El registro queda en `results/instancias/cola_terna.log`.

## Qué deja cada corrida (`results/instancias/corridas/<modelo>_s<semilla>/`)

- `resumen.json`: tiempo, memoria GPU, parámetros, entorno y métricas.
- `metricas.json`: AP, AP50, AP75 y AP_small de máscara en val y en test; PQ, SQ y RQ por imagen; PQ por grupo; umbral de score.
- `test_por_imagen.csv`: una fila por imagen de test con grupo, TP, FP, FN, PQ, SQ, RQ y Dice semántico. Es la entrada de la sección 5 (estadística por grupo).
- `preds_val.json` y `preds_test.json`: predicciones en formato de resultados COCO (RLE), para recalcular cualquier métrica sin reentrenar.

## Protocolo (igual para los modelos)

- Partición agrupada congelada `results/eda/particion.csv` (columna `split`): 744/93/108 imágenes, 14 grupos en test.
- Pesos preentrenados en COCO. Imagen completa 1280×720, sin reescalar ni recortar, porque el diámetro mediano del tornillo es 10.6 px (EDA de instancias, sección 5). Volteo horizontal como aumento.
- Mask R-CNN:
  - ResNet-50-FPN.
  - Anclas de 8–128 px, porque las de torchvision por defecto van de 32 a 512 y el 98,8 % de los tornillos son *small* según COCO (menos de 32×32 px).
  - SGD con momento 0.9, calentamiento lineal y coseno, precisión mixta.
  - Mejor época por AP de máscara en val.
- YOLOv8-seg:
  - Variante n (3.26 M parámetros), imgsz 1280, lote 4.
  - Resto de hiperparámetros por defecto de ultralytics.
  - Mejor época por el *fitness* de ultralytics en val.
- Métricas:
  - El AP de máscara se calcula sin umbral.
  - El PQ usa un umbral de score elegido en val (el que maximiza el PQ medio por imagen) y se aplica sin cambios en test.
  - Test se evalúa una sola vez por corrida.
- Dos anotaciones tienen más de un polígono. En el formato YOLO se usa el polígono más grande; en COCO (Mask R-CNN y evaluación) se usan todos.

## SOLOv2 (entorno aparte, `wdsd_mmdet`)

MMDetection 3.3 necesita mmcv 2.1, que tiene ruedas precompiladas para Windows solo con torch 2.1 y Python ≤ 3.11. Por eso va en un entorno aparte, para no tocar `wdsd_dl`. La instalación no usa la GPU, así que se puede hacer mientras corre el piloto.

```powershell
& C:\Users\ivanj\miniconda3\Scripts\conda.exe create -n wdsd_mmdet python=3.11 -y
$PYM = "C:\Users\ivanj\miniconda3\envs\wdsd_mmdet\python.exe"
& $PYM -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
& $PYM -m pip install "numpy<2" mmengine==0.10.4 pycocotools pandas scipy
& $PYM -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
& $PYM -m pip install mmdet==3.3.0
& $PYM -m pip install "setuptools<70"   # torch 2.1 usa pkg_resources, que setuptools nuevo ya no trae
& $PYM -c "import torch, mmcv, mmdet; from mmcv.ops import nms; print(torch.cuda.is_available(), mmcv.__version__, mmdet.__version__)"
```

- La última línea debe imprimir `True 2.1.0 3.3.0`.
- Si pip empieza a **compilar** mmcv (aparece `Building wheel for mmcv` y tarda), cancelar con Ctrl+C: significa que no encontró la rueda para esa combinación. En ese caso, avisar.

Piloto y cola (requiere haber corrido antes `entrenar_instancias.py preparar`):

```powershell
& $PYM solov2_instancias.py piloto
& $PYM solov2_instancias.py cola --hasta 07:30 --epocas 12 --semillas 42,43,44,45,46
```

- Configuración oficial `solov2_r50_fpn_1x_coco` con pesos COCO (34.8 AP de máscara en COCO) y 1 clase.
- Lote 2 con precisión mixta. Si falta memoria, usar `--lote 1`.
- Deja los mismos archivos que los otros modelos en `results/instancias/corridas/solov2_s<semilla>/`, así que `resumen` y `estadistica_instancias.py` lo incluyen solo.
