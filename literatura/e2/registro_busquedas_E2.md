# Registro de búsquedas del E2 (ejecutadas por Ivan, 2026-11-15)

Fecha de ejecución: 2026-11-15 (hora de descarga de los archivos: 17:23–17:37, hora de Colombia). Cadenas exactas: `cadenas_busqueda_E2.md`. Los archivos exportados no se versionan (términos de uso de Scopus y WoS); `cribado_e2.py` los lee de `exportaciones/` y reproduce todos los conteos de este registro.

| Cadena | Scopus | WoS | IEEE Xplore | Total |
|---|---|---|---|---|
| A1 objetos pequeños y sujetadores | 108 | 55 | 100 (¿tope de exportación? por confirmar) | 263 |
| A2 inspección industrial y desmontaje | 589 | 283 | — | 872 |
| A3 surveys y benchmarks | 56 | 34 | — | 90 |
| B1 asignación de etiquetas, objetos diminutos | 34 | — | — | 34 |
| B2 convolución dinámica | 126 | — | — | 126 |
| B3 atención por consultas / transformers de máscara | 38 | — | — | 38 |
| B4 contexto objeto–entorno | 80 | — | — | 80 |
| **Total** | 1031 | 372 | 100 | **1503** |

## PRISMA, fase de identificación (calculado 2026-11-15)
- Registros identificados: 1503.
- Duplicados eliminados: 440. Regla: mismo DOI normalizado; si falta el DOI, mismo título normalizado (minúsculas, sin signos).
- Registros únicos: 1063 (capa A: 835; capa B: 230; 2 en ambas).
- Excluidos automáticamente por tipo de documento (no son estudios): 38 (35 "Conference review" = portadas de actas, 3 "Erratum").
- **Pasan a cribado por título y resumen: 1025** (capa A: 811; capa B: 215; 1 en ambas).

Archivo de cribado: `registros_unicos_E2.csv` (lo genera `cribado_e2.py`). Las columnas `menciona_instancias`, `reporta_metrica`, `es_3d` y `arquitecturas_mencionadas` son marcas automáticas por palabras clave en título y resumen: ayudan a ordenar, **no son decisiones**.

## Criterios de cribado por título y resumen (propuesta, fijar antes de empezar)
**Capa A (problema). Incluir si cumple las tres:**
1. Usa Deep Learning para segmentación de instancias (una máscara por objeto) o panóptica, en imágenes 2D.
2. Reporta al menos una métrica cuantitativa (AP, mAP, F1, IoU, etc.).
3. Es artículo o ponencia revisada por pares (los surveys de A3 se incluyen como fuente de arquitecturas).

**Excluir, con una de estas razones (para el diagrama PRISMA):** solo detección con cajas; solo segmentación semántica; nubes de puntos o 3D; no usa Deep Learning; no es un estudio; sin métricas.

**Capa B (mecanismos). Incluir si** describe el mecanismo buscado (asignación de etiquetas, convolución dinámica, consultas/atención de máscara, contexto) aplicado en una modalidad distinta a la nuestra.

Cada decisión: `incluir`, `excluir` (con razón) o `dudoso`. Los dudosos se resuelven por consenso y se registra cómo.

## Cribado de la capa A (2026-11-15, asistido por IA; reemplaza la propuesta de dos revisores)
Reglas aplicadas en orden (`cribado_e2.py`), sobre los 835 registros únicos de la capa A:

| Regla | Excluye |
|---|---|
| R0 tipo de documento (portadas de actas, erratas) | 24 |
| R1 3D o nube de puntos | 99 |
| R2 no menciona segmentación de instancias en título/resumen | 51 |
| R3 no reporta métricas en el resumen | 158 |
| R4 no nombra ≥ 2 arquitecturas de segmentación de instancias en el resumen (y no es survey/benchmark de A3) | 423 |
| R5 lectura del resumen por la IA (80 leídos) | 6 |
| **Incluidos** | **62** |
| **Dudosos (verificación humana)** | **12** |

Comprobación: 24 + 99 + 51 + 158 + 423 + 6 + 62 + 12 = 835.

**Limitación medida de R4:** en una muestra al azar de 30 registros excluidos por R4 (semilla 42), 5 comparan claramente varias arquitecturas sin nombrarlas en el resumen y 3 más quizá. Tasa estimada de pérdida: 5/30 a 8/30 = 17 % a 27 %, o sea, unos 70–110 de los 423. Mitigación: la lista larga de arquitecturas sale sobre todo de los surveys y benchmarks (A3), no de estos estudios.

Archivo: `registros_unicos_E2.csv` (columnas `regla` y `decision_final`); las lecturas de R5 y la resolución de dudosos están en `decisiones_lectura_E2.csv` (columna `evidencia_texto_completo`).

## Resolución de los 12 dudosos (2026-11-15, Ivan + contraste con texto completo)
- Incluidos: R0295 (YOLOv8n-seg vs YOLO11n-seg) y R0255 (DPR-GAN; Mask R-CNN y YOLACT con mAP de máscara).
- Excluidos por texto completo no recuperado: R0027, R0684.
- Excluidos por una sola arquitectura de segmentación de instancias: R0170, R0251, R0364, R0496, R0792, R0812.
- Excluido porque la tarea es semántica: R0228.
- Excluido por supervisión débil (solo cajas): R0795. **Aclaración de criterio añadida durante el cribado:** se excluyen los métodos entrenados sin máscaras (supervisión débil), porque su desempeño no es comparable con el de modelos entrenados con máscaras.
- Ivan propuso incluir R0027 y R0812, y excluir R0255. Tras contrastar con el texto, aceptó los tres cambios.

**Total final de la capa A: 64 incluidos** (62 + 2).

## Puntaje provisional del Top 10 (2026-11-15, 19:30 COL)
- Citas: Scopus (CSV de Ivan, 39 registros: 21 originales + 18 aplicaciones de "Mask Scoring R-CNN" descartadas). Año = año de Scopus (Cascade TPAMI 2021, YOLOv9 ECCV → 2025).
- Faltan citas de Google Scholar: YOLOv8, YOLO11, YOLO26 (software) y RTMDet (arXiv).
- Regla de configuración para el AP: la configuración de referencia más pequeña publicada con pesos COCO (R50 en MMDetection; "n" en ultralytics; yolov7-seg y gelan-c-seg, que son las únicas publicadas en sus repos).
- Decisión de método a declarar: las citas de YOLOv7-seg y YOLOv9-seg son las del paper que define la arquitectura (paper de detección); la cabeza de segmentación no tiene paper propio.
- Google Scholar (Ivan, 2026-11-15): "Ultralytics yolo" (Zenodo 2023) 6828 citas, registro compartido por YOLOv8 y YOLO11 (YOLO11 no tiene paper; ambos reciben ese registro); "Ultralytics YOLO26" (arXiv 2026) 389; RTMDet (arXiv 2212.07784) 1598.
- Regla: citas por año con el año del registro citado; recencia con el año de lanzamiento del modelo (YOLO11 = 2024).
- Hoja `puntaje_calculado` de lista_larga_puntaje_E2.xlsx = puntaje vigente.

## Cribado de la capa B (2026-11-15, asistido por IA, por título)
214 registros cribados (solo capa B, sin los excluidos por tipo de documento); 56 incluidos con su mecanismo: M1 14, M2 2, M3 4, M4 21, M5 1, M6 14. Detalle y transferibilidad en `tabla_mecanismos_E2.md`.

## Puntaje final y Top 10 (cerrado 2026-11-16)
`puntaje_e2.py` calcula el puntaje desde `lista_larga_E2.csv` con las reglas de la hoja `reglas` de `lista_larga_puntaje_E2.xlsx` (fijadas antes de calcular) y da los mismos valores que la hoja `puntaje_calculado`.

- F4 (cabe en 6 GB) se verificó con la prueba de humo de 300 actualizaciones de pesos (`results/e2/prueba_humo/`), más exigente que la "prueba de 1 iteración" escrita en la hoja `reglas`.
- Mask DINO (puesto 6 por puntaje) no pasa F4: su README declara 15 GB por GPU con R50 y solo existe en detectron2.
- Los 10 de mayor puntaje que pasan los filtros cubrían 3 familias (Q, P, R). Regla de familias: sale RTMDet-Ins (P, 0.619) y entra CondInst (K, 0.469).
- **Top 10:** Mask2Former, YOLOv7-seg, YOLOv9-seg, YOLO26-seg, YOLO11-seg, Cascade Mask R-CNN, HTC, PointRend, Mask Scoring R-CNN, CondInst. **Reservas:** QueryInst, SCNet.

## Modelos muy citados que quedan fuera del Top 10 (PDF E2 §3.2)
Se listan los que tienen más citas por año que CondInst (86.0), el modelo del Top 10 con menos citas por año. Valores de `puntaje_E2.csv`.

| Modelo | Familia | Citas por año | Puntaje | Razón |
|---|---|---|---|---|
| Mask R-CNN, YOLOv8-seg, SOLOv2 | R, P, K | 2651.1, 1707.0, 156.3 | 0.673, 0.605, 0.599 | Modelos base del E1: van además del Top 10 (§3.2). |
| SAM | S | 2640.0 | – | No pasa F1: necesita prompts (puntos o cajas) y no da una máscara por instancia por sí solo. |
| RTMDet-Ins | P | 319.6 | 0.619 | Estaba entre los 10 primeros, pero sale por la regla de 4 familias: es el último de la familia P, la más representada, y entra CondInst (K). |
| YOLACT | P | 313.1 | 0.465 | Puntaje menor que los 10 elegidos. Su AP de máscara en COCO es el más bajo de la lista larga. |
| Mask DINO | Q | 154.3 | 0.629 | No pasa F4: su README declara 15 GB por GPU con R50, y solo existe en detectron2. |
| CenterMask | P | 104.6 | 0.417 | Puntaje menor que los 10 elegidos. |
| SOLO | K | 104.0 | 0.458 | Puntaje menor que los 10 elegidos. Además, descarta objetos menores que la celda del nivel; inferido del código. |
| BlendMask | P | 93.1 | 0.387 | Puntaje menor que los 10 elegidos. |
| PolarMask | C | 90.3 | 0.355 | Puntaje menor que los 10 elegidos. |
