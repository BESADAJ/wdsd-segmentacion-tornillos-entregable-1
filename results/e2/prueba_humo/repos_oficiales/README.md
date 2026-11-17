# Prueba de humo de YOLOv7-seg y YOLOv9-seg (repositorios oficiales)

Estos dos modelos no están en MMDetection ni en ultralytics. Se corren con los repositorios de sus autores, con el mismo protocolo que `prueba_humo_e2.py`:

- las mismas 4 imágenes, preparadas con `prueba_humo_e2.py datos`;
- 1280 px y AMP;
- 300 actualizaciones de pesos;
- sin aumentación.

## Entorno

`wdsd_yolo9` es un clon de `wdsd_mmdet` (torch 2.1.2 + CUDA 12.1). Se le hicieron tres cambios:

- numpy 1.26.4, porque torch 2.1.2 no funciona con NumPy 2.
- opencv-python 4.11.0.86.
- Se quitó albumentations: instalaba opencv-python-headless y NumPy 2.

`PYTHONUTF8=1` es necesario porque la ruta tiene "í".

Versiones exactas: `requirements_wdsd_yolo9_congelado.txt` (este entorno) y `requirements_wdsd_mmdet_congelado.txt` (MMDetection), en la raíz del repo.

| Repo | Commit / rama | Pesos COCO |
|---|---|---|
| https://github.com/WongKinYiu/yolov9 | `5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff` (`main`) | `gelan-c-seg.pt` (release v0.1) |
| https://github.com/WongKinYiu/yolov7 | `44f30af0daccb1a3baecc5d80eae22948516c579` (rama `u7`, carpeta `seg/`) | `yolov7-seg.pt` (release v0.1) |

## Cambios a los repositorios (declarados)

1. **`segment/train.py`, en los dos repos:** `nbs = 64` pasa a `nbs = 2`.
   - Por defecto acumulan gradientes hasta un lote nominal de 64.
   - Con lote 2 eso da un paso del optimizador cada 32 iteraciones, o sea unos 10 pasos en 300 iteraciones.
   - Con `nbs = 2`, lote 2 da 1 paso por iteración, y lote 1 da 1 paso cada 2 iteraciones.
   - Así, 150 épocas sobre 4 imágenes dan 300 pasos en ambos casos.
2. **`data/hyps/hyp.humo.yaml`, en los dos repos:** es el `hyp.scratch-high.yaml` de cada repo con toda la aumentación en 0 (mosaic, mixup, copy_paste, scale, translate, degrees, shear, perspective, hsv, flips). Copias: `yolov7_hyp.humo.yaml` y `yolov9_hyp.humo.yaml`.
3. **Solo YOLOv7 u7:**
   - `np.int`, `np.float`, `np.bool` y `np.object` pasan a los tipos de Python, porque NumPy ≥ 1.24 los eliminó.
   - Se quita `return_seg=True,` en la llamada a `random_perspective` de `utils/segment/dataloaders.py`. Es un error del repo: esa función no acepta ese argumento, y la ruta sin mosaico falla.
4. **Solo YOLOv9:** el mismo reemplazo de `np.int` y similares, donde aparece.
5. **Solo YOLOv9, añadido para el E2 (2026-11-16):** en `utils/segment/loss_tal.py`, `target_scores_sum = target_scores.sum()` pasa a `target_scores_sum = max(target_scores.sum(), 1)`, igual que en la pérdida de ultralytics. Sin el cambio, cuando el asignador da puntaje 0 a todos los objetivos de una imagen, las pérdidas de clase y de caja se dividen por 0. En el piloto del E2 en la 3060 eso dio una pérdida media infinita en la primera época. Copia del archivo parchado: `yolov9_loss_tal.py`.

## Comandos (PowerShell, desde la carpeta de cada repo)

```powershell
$env:PYTHONUTF8 = "1"
$HUMO = "<repo E2>\datos_inst\humo_yolo\data.yaml"
# YOLOv9 (yolov9/)
& $PY9 segment/train.py --data $HUMO --cfg models/segment/gelan-c-seg.yaml --weights gelan-c-seg.pt --hyp data/hyps/hyp.humo.yaml --img 1280 --batch 1 --epochs 150 --device 0 --workers 0 --name humo_b1 --exist-ok
& $PY9 segment/val.py --data $HUMO --weights runs/train-seg/humo_b1/weights/last.pt --img 1280 --batch 1 --task train --device 0
# YOLOv7 (yolov7u7/seg/): lote 2 y lote 1
& $PY9 segment/train.py --data $HUMO --cfg models/segment/yolov7-seg.yaml --weights yolov7-seg.pt --hyp data/hyps/hyp.humo.yaml --img 1280 --batch 2 --epochs 150 --device 0 --workers 0 --name humo_b2 --exist-ok
& $PY9 segment/train.py --data $HUMO --cfg models/segment/yolov7-seg.yaml --weights yolov7-seg.pt --hyp data/hyps/hyp.humo.yaml --img 1280 --batch 1 --epochs 150 --device 0 --workers 0 --name humo_b1 --exist-ok
```

## Memoria: cómo se detecta que no cabe

Estos repos no tienen el tope de VRAM de `prueba_humo_e2.py`. En Windows, cuando se acaba la VRAM, el driver usa RAM del sistema sin dar error.

- La señal es la columna `GPU_mem` del log. Es la memoria reservada por PyTorch, en GB de 10^9 bytes.
- La tarjeta tiene 6144 MiB = 6.44 GB. Un valor por encima de 6.44 GB es desborde a RAM, aunque el log no muestre error.
- Otra señal es el tiempo: YOLOv7 con lote 2 tardó 24 min.

## Resultados (logs en esta carpeta, convertidos a UTF-8)

| Modelo | Lote | GPU_mem pico | AP50 máscara (4 img) | Log |
|---|---|---|---|---|
| YOLOv7-seg | 2 | 7.41 GB → desborda | 0.70 | `yolov7_b2_train.log` |
| YOLOv7-seg | 1 | 3.89 GB | **0.82** (caja 0.995) | `yolov7_b1_train.log` |
| YOLOv9-seg | 2 | 6.89 GB → desborda | no aplica* | `yolov9_b2_train_scratch-high.log`, `yolov9_b2_val.log` |
| YOLOv9-seg | 1 | 3.57 GB | **0.79** con el mejor checkpoint; 0.69 con `last.pt` (caja 0.92) | `yolov9_b1_train.log`, `yolov9_b1_val.log` |

\* La corrida de YOLOv9 con lote 2 usó `hyp.scratch-high.yaml`, con aumentación, antes de crear `hyp.humo.yaml`. Vale solo para la memoria, porque la aumentación no cambia el tamaño de entrada (1280 px). Su AP no se usa.

El AP50 de la tabla sale de la validación final que hace `train.py` sobre las mismas 4 imágenes (train = val en `data.yaml`).

La columna "M" es la de máscara y la "B" la de caja. Ejemplo de la última línea `all` de `yolov7_b1_train.log`:

- columnas de caja (B): P 1.0, R 0.994, mAP50 0.995, mAP50-95 0.869;
- columnas de máscara (M): P 0.846, R 0.843, mAP50 **0.817**, mAP50-95 0.325.
