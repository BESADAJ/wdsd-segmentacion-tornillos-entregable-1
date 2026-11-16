# Prueba de humo del E2: resultado final (2026-11-15/16, RTX 3060 Laptop 6 GB)

Protocolo: 4 imágenes de train (13 tornillos), 1280x720, AMP, lote 2 y si no cabe lote 1; evaluación en las mismas 4 imágenes (AP50 de máscara, mejor checkpoint). 300 actualizaciones de pesos (900 para los que no llegaron a 0.80 en 300, desviación declarada). Sin aumentación.
Script: `prueba_humo_e2.py` (raíz del repo; los comandos exactos están en su docstring). Fuentes: `tabla_v2.csv` (MMDetection y ultralytics) y los logs de `repos_oficiales/` (YOLOv7 y YOLOv9).

| Puesto | Modelo | Familia | Lote | Memoria pico | AP50 máscara | AP50 caja | Sanidad ≥ 0.80 |
|---|---|---|---|---|---|---|---|
| 1 | Mask2Former | Q | 1 (lote 2 sin memoria) | 3.19 GB (5.47 reservada) | 0.84 (900 pasos) | – | sí |
| 2 | YOLOv7-seg | P | 1 (lote 2: 7.41 GB, desborde) | 3.89 GB | 0.82 | 0.995 | sí |
| 3 | YOLOv9-seg (GELAN-C) | P | 1 (lote 2: 6.89 GB, desborde) | 3.57 GB | 0.79 | 0.92 | no (techo de máscara) |
| 4 | YOLO26n-seg | P | 2 | 1.72 GB | 0.76 | 0.995 | no (techo de máscara) |
| 5 | YOLO11n-seg | P | 2 | 1.34 GB | 0.70–0.72 | – | no (techo de máscara) |
| 6 | Cascade Mask R-CNN | R | 2 | 2.25 GB | 1.00 | – | sí |
| 7 | HTC | R | 2 | 2.44 GB | 1.00 | – | sí |
| 8 | PointRend | R | 2 | 1.98 GB | 1.00 | – | sí |
| 9 | Mask Scoring R-CNN | R | 2 | 1.94 GB | 0.97 | – | sí |
| 10 | CondInst | K | 2 | 2.16 GB | 0.48–0.54 | – | no |
| Reserva 1 | QueryInst | Q | 2 | 3.80 GB | 1.00 | – | sí |
| Reserva 2 | SCNet | R | 2 | 2.37 GB | 1.00 | – | sí |
| Control | YOLOv8n-seg (base E1) | P | 2 | 1.18 GB | 0.73 | – | no |

Fuera: Mask DINO (15 GB documentados, F4). SOLO: descarta máscaras menores que ~el paso del nivel; no ve tornillos de ~10 px (inferido del código).

Selección (reglas fijadas antes del puntaje, hoja `reglas`): los 10 de mayor puntaje que pasan F1–F4; como cubrían 3 familias (Q, P, R), RTMDet (último de P) se sustituye por CondInst (mejor de K). La sanidad no es filtro: se reporta (PDF E2 §11.1). Las familias P y K arman la máscara a 1/4 de resolución y quedan en ~0.70–0.80 de AP50 de máscara con caja ~1.0, igual que el control YOLOv8n-seg que funciona en el E1.

Ajustes declarados respecto a las configuraciones oficiales:
- MMDetection con RPN: anclas 8–128 px (escala 2), como la Mask R-CNN del E1 (oficial 32–512 px no alcanza IoU 0.3 con tornillos de ~9x8 px).
- ultralytics: nbs = lote; YOLOv7/v9: `nbs = 2` en segment/train.py (por defecto acumulan hasta 64 → ~10 actualizaciones en 300 iteraciones).
- Tope de VRAM 95 % en el script propio (Windows desborda a RAM sin error); en YOLOv7/v9 se detecta por GPU_mem > 6.44 GB y tiempo.
- YOLOv7 u7: error del repo en la ruta sin mosaico (`return_seg=True`), parcheado; `np.int` → `int` (NumPy ≥ 1.24).
- Entorno wdsd_yolo9 (clon de wdsd_mmdet): numpy 1.26.4, opencv-python 4.11.0.86, sin albumentations; `PYTHONUTF8=1` por la ruta con "í".

## Archivos

- `tabla_v2.csv`: una fila por intento, en orden de fecha. Las filas que valen para cada modelo:
  - **YOLO11 y YOLO26:**
    - Las filas de las 22:41–22:45 tienen AP 0 por la acumulación de gradientes (`nbs=64`), corregida en el v2. No valen.
    - Las de las 23:03–23:06 son la corrida de 300 pasos.
    - Las de las 23:52 y 00:01 son la de 900 pasos (`--epocas 450`).
    - Las de las 00:47 y 00:50 repiten la de 300 pasos junto al control y dan el mismo resultado.
  - **Mask2Former:**
    - 23:29, lote 2: sin memoria.
    - 23:32, lote 1, 300 pasos: AP50 0.16.
    - 00:38, lote 1, 900 pasos: AP50 0.84.
  - **CondInst:** 23:29 son 300 pasos; 00:07 son 900 pasos.
  - **Control YOLOv8n-seg:** fila de las 00:45.
- `tabla_v1_descartada.csv`: primera corrida, con el script v1. No vale ni para memoria ni para sanidad, por las tres fallas que corrige el v2 (ver el docstring del script). Se guarda para documentar por qué se repitió. SOLO solo tiene esta corrida.
- `<modelo>/resumen.json`: última corrida de cada modelo. `<modelo>/lote<N>/`: configuración usada y log de MMDetection o `args.yaml` y `results.csv` de ultralytics. Los pesos no se versionan.
- `repos_oficiales/`: comandos, parches y logs de YOLOv7-seg y YOLOv9-seg.
