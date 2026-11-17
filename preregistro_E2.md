# Preregistro del Entregable 2

Estado: **PREREGISTRO, 2026-11-16 (hora de Colombia).** Cuenta como preregistro solo cuando esté fechado en una release etiquetada (`e2-preregistro`) **antes del primer entrenamiento** del E2 (PDF E2 §9). Antes de la release solo se corrió el piloto de tiempo (§4: 1 época por modelo en la 3060, sin guardar ni usar métricas). La prueba de humo de `results/e2/prueba_humo/` usó solo 4 imágenes de train, para medir memoria y sanidad; no usó validación ni prueba.

Cada campo dice de dónde viene:
- **[E1]**: heredado sin cambios de `results/eda/declaracion_a_priori_instancias.json` (2026-11-11), como exige el PDF E2.
- **[E2]**: decisión nueva de este documento.

## 1. Tarea, datos y congelamiento

| Campo | Valor |
|---|---|
| Tipo de tarea | 8. Segmentación de imágenes, variante de instancias **[E1]**. |
| Unidad de independencia | Grupo de escena (columna `grupo` de `results/eda/particion.csv`) **[E1]**. Test tiene 14 grupos. |
| Partición | 744 / 93 / 108 imágenes (3530 / 479 / 389 tornillos) **[E1]**, sin modificar. |
| Hash SHA-256 del dataset | `39117b86f77b4f4277045d904848291306b4492bcd553872d6dddb097c4544a4` **[E1]** |
| Hash de train | `efb6e322326e5d935919045586c4169155cb2fb0f7d44294f72eb7efa2b984c4` **[E1]** |
| Hash de val | `8252f051f632fa44a1be3f1e39146d0b35422232b986043003caa73b3d2b9166` **[E1]** |
| Hash de test | `af1fed4e311dfab3fe2762961b6d6b5bbd6c36184ff179af77247a76e16e80fa` **[E1]** |
| Entrada | Imagen completa de 1280×720, sin ventana deslizante **[E1]**. |
| Código y entornos | El commit de la release `e2-preregistro`; `requirements_wdsd_dl_congelado.txt`, `requirements_wdsd_mmdet_congelado.txt` y `requirements_wdsd_yolo9_congelado.txt`; YOLOv9 `5b1ea9a`, YOLOv7 u7 `44f30af`, con los cambios declarados en `results/e2/prueba_humo/repos_oficiales/README.md` **[E2]**. |

## 2. Métricas y márgenes

| Campo | Valor |
|---|---|
| Métrica primaria | F1 de instancia por imagen: emparejamiento húngaro por IoU de máscara, umbral 0.5. Para la inferencia se promedia por grupo **[E1]**. |
| Secundarias | Ninguna sustenta conclusiones confirmatorias **[E1]**: SQ y PQ a IoU 0.5, AP50 de máscara, error absoluto en el conteo de tornillos por imagen, Dice e IoU de píxel, exactitud de píxel y mIoU, F1 por familia de escena. |
| Informativas | AP 0.5:0.95 de máscara y F1 a IoU 0.75 **[E1]**. |
| Posproceso | Umbral de score elegido en val, área mínima 5 px, conectividad 8 **[E1]**. |
| δ (relevancia práctica) | **0.20** de F1 **[E1]**. |
| α | 0.05 **[E1]**. |
| Techo de anotación | F1 a IoU 0.5 = 0.9995; a IoU 0.75 = 0.9063 **[E1]**. |
| δg (sobreajuste) | **0.15** de F1 entre train y val **[E2]**. El F1 de train se mide en una submuestra fija de 93 imágenes de train (`random.Random(0)`, `entrenar_e2.py datos`), con el mismo pipeline y el umbral de score de val. El E1 no midió esta brecha, así que no hay evidencia previa para el valor. |

## 3. Modelos

**Top 10 [E2].** Selección cerrada el 2026-11-16 con `literatura/e2/puntaje_e2.py`.

| # | Modelo | Familia | Marco | Configuración base | Lote |
|---|---|---|---|---|---|
| 1 | Mask2Former R50 | Q, consultas | MMDetection 3.3 | `mask2former_r50_8xb2-lsj-50e_coco` | 1 |
| 2 | YOLOv7-seg | P, prototipos | repo oficial u7 | `yolov7-seg.yaml` | 1 |
| 3 | YOLOv9-seg (GELAN-C) | P | repo oficial | `gelan-c-seg.yaml` | 1 |
| 4 | YOLO26n-seg | P | ultralytics 8.4.207 | `yolo26n-seg` | 2 |
| 5 | YOLO11n-seg | P | ultralytics 8.4.207 | `yolo11n-seg` | 2 |
| 6 | Cascade Mask R-CNN R50 | R, dos etapas | MMDetection 3.3 | `cascade-mask-rcnn_r50_fpn_1x_coco` | 2 |
| 7 | HTC R50 | R | MMDetection 3.3 | `htc-without-semantic_r50_fpn_1x_coco` | 2 |
| 8 | PointRend R50 | R | MMDetection 3.3 | `point-rend_r50-caffe_fpn_ms-1x_coco` | 2 |
| 9 | Mask Scoring R-CNN R50 | R | MMDetection 3.3 | `ms-rcnn_r50-caffe_fpn_1x_coco` | 2 |
| 10 | CondInst R50 | K, kernels | MMDetection 3.3 | `condinst_r50_fpn_ms-poly-90k_coco_instance` | 2 |

- **Reservas, en orden:** QueryInst y luego SCNet. Una reserva entra solo si un modelo falla técnicamente en las 5 semillas (regla de exclusión, §6). Nunca entra por desempeño.
- **Modelos base del E1, además del Top 10 [E1]:** Mask R-CNN R50 (torchvision), YOLOv8n-seg (ultralytics) y SOLOv2 R50 (MMDetection). Se reentrenan con el mismo presupuesto que el Top 10 (PDF Apéndice B).

**Baselines [E2].**
- **Trivial:** máscara vacía, sin detecciones (F1 = 0). Es un filtro, no un contraste: un modelo que no lo supera se reporta como fallido (PDF §7.5).
- **No profundo:** regresión logística por píxel sobre 15 rasgos: RGB y HSV, y sobre el gris un filtro gaussiano, la magnitud del gradiente y el laplaciano con σ = 1, 2 y 4 px. Se entrena con todos los píxeles de tornillo de train y 1000 de fondo por imagen, con las dos clases con el mismo peso. Las instancias son las componentes conexas (conectividad 8, área ≥ 5 px) de la probabilidad > 0.5, con score = probabilidad media; son válidas aquí porque en el EDA ningún tornillo toca a otro. Mismo número T = 3 de configuraciones: λ (regularización L2) ∈ {1e-4, 1e-3, 1e-2}. La semilla cambia la muestra de píxeles de fondo. Código: `src/wdsd_e2/m_clasico.py`.

**Ajustes comunes a todos [E2], los mismos de la prueba de humo:**
- Pesos preentrenados en COCO train2017 (118 mil imágenes), de la fuente oficial de cada marco.
- AMP y **lote efectivo 2** para todos: lote 2, o lote 1 con acumulación de 2 iteraciones por paso del optimizador en los que no caben en 6 GB (Mask2Former: `accumulative_counts = 2`; YOLOv7 y YOLOv9: `nbs = 2` con lote 1). Así todos dan el mismo número de pasos por época (372). Con lote 1 las estadísticas de BatchNorm son más ruidosas; se declara como limitación.
- Modelos con RPN: anclas de 8 a 128 px (escala 2).
- ultralytics: `nbs = 2` y el optimizador que elige `optimizer=auto` con 1 clase y menos de 10 000 pasos (AdamW, lr 0.002), fijado para que no cambie con las épocas.
- **Programa común:** calentamiento lineal de min(500, pasos totales / 10) pasos y después coseno hasta el 1 % de la tasa inicial (por iteración en MMDetection y torchvision, por época en los YOLO, que es como lo implementan).
- **Tasa por defecto de cada marco**, que es el centro de la búsqueda: SGD de MMDetection = la oficial × 2/16 (lote 16 → 2); AdamW de MMDetection (Mask2Former, QueryInst) = la oficial sin escalar; ultralytics = 0.002 (AdamW); YOLOv7 y YOLOv9 = 0.01 (SGD, su `hyp.scratch-high.yaml`); Mask R-CNN de torchvision = 0.005.
- Única aumentación: volteo horizontal con p = 0.5, igual para todos. Mosaic, mixup, HSV y escala quedan apagados. Esto corrige una diferencia del E1, donde YOLOv8 usó la aumentación por defecto de ultralytics.
- Mejor época elegida por AP de máscara 0.5:0.95 en val, como en el E1 (no el *fitness* de ultralytics ni de los repos YOLO, que mezcla caja y máscara). Test una sola vez por corrida.
- Inferencia igual para todos: score ≥ 0.05, como mucho 100 detecciones por imagen, NMS 0.6 en los YOLO.
- Código: `entrenar_e2.py` y `src/wdsd_e2/`. Cada corrida deja `preds_*.json`, `historial.csv` (pérdida y AP de val por época), `metricas.json` y `resumen.json`.

## 4. Presupuesto de entrenamiento y ajuste

**Fijo [E2].** Mismo presupuesto para los 13 modelos (Top 10 y modelos base) y el baseline no profundo (PDF §11.1 y §11.2):

- **Búsqueda de hiperparámetros:** T = 3 configuraciones por modelo. Son la tasa de aprendizaje por defecto de cada marco multiplicada por {0.5, 1, 2}, en escala logarítmica.
  - Cada configuración se entrena con la semilla 42 y ⌈E/3⌉ épocas (2 con E = 6).
  - Se elige por el F1 de instancia promediado por grupo en val, con el umbral de score elegido en val. Si la mejor supera a la de por defecto por menos de 0.005, gana la de por defecto.
  - Se reportan el mejor valor, su posición en el espacio y la curva de F1 contra tasa de aprendizaje.
- **Corridas finales:** E épocas iguales para todos, con la configuración elegida y R = 5 semillas. **E = 6**, fijado el 2026-11-16 con el piloto y la regla de abajo (el valor inicial propuesto era 12, el programa "1x" de MMDetection).
- **Si una configuración de la búsqueda falla dos veces**, se elige entre las que terminaron. Si fallan las tres, el modelo no tiene corridas finales y se reporta como fallo.
- **Reparto entre máquinas por semillas, nunca por modelos (como en el E1):** cada modelo tiene las mismas semillas en cada GPU, así que el hardware queda balanceado entre modelos. El tiempo de entrenamiento se reporta por máquina. La latencia de inferencia se mide para todos en una sola GPU. Las T configuraciones de la búsqueda de un modelo van en la misma máquina.
- **Piloto de tiempo:** 1 época por modelo en cada máquina, semilla 42, solo para medir segundos por época, tiempo fijo y segundos de predicción por imagen. No predice test ni calcula F1; el AP de val que imprimen los marcos durante la época no se guarda ni se usa.
- **Si el hardware no alcanza:** se baja E, igual para todos, **antes** de la release. Nunca después de ver resultados.
- **Parada:** el mismo criterio para todos: E épocas fijas y la mejor época por val. Se reportan las curvas de train y val. Si la curva de val de un modelo sigue subiendo en la última época, se declara que ese modelo quedó subentrenado con E.

**Hardware [E2], confirmado el 2026-11-16:** dos portátiles, sin nube.
- RTX 3060 Laptop de 6 GB: piloto, búsqueda de hiperparámetros y semillas 42–45. Desde la noche del 2026-11-16.
- Semilla 46: entera en la 3060 o entera en la RTX 3050 Laptop de 6 GB (disponible del 2026-11-17 20:00 al 2026-11-18 06:00). Se decide antes de que empiece la semilla 46 y se reporta. Nunca se reparten los modelos de una misma semilla entre máquinas, salvo para terminar corridas que la 3050 no alcance a las 06:00, lo que se declararía.
- El tiempo en la 3050 se estima con el piloto de la 3060 por la razón 3050/3060 que midió el E1 en cada marco (`results/instancias/corridas/*/resumen.json`, minutos por corrida): MMDetection 1.76 (SOLOv2: 55.9 / 31.7), torchvision 2.19 (Mask R-CNN: 73.5 / 33.5), ultralytics 2.97 (YOLOv8: 62.4 / 21.0). YOLOv7 y YOLOv9 usan 2.97, la más lenta.

**Regla para fijar E**, decidida antes de entrenar:
1. El piloto de tiempo mide en la 3060 los segundos por época, el tiempo fijo y los segundos de predicción por imagen de cada modelo (`entrenar_e2.py piloto`, `results/e2/piloto/3060/`).
2. Con esa medida se elige el E más grande, entre 12 y 4, con el que la búsqueda y las 5 semillas caben antes de las 06:00 del 2026-11-18, para dejar el resto del día al análisis (entrega: 2026-11-18, 24:00).
3. Ese E es el mismo para todos los modelos. Se anota aquí, con la fecha, antes de publicar la release.

**Piloto (3060, observado, 2026-11-16).** Minutos por época con la validación incluida: Mask2Former 7.5, YOLOv7 4.7, YOLOv9 4.6, HTC 4.2, Cascade 3.7, PointRend 3.1, CondInst 3.0, MS R-CNN 3.0, Mask R-CNN 3.0, SOLOv2 2.8, YOLO26 2.1, YOLO11 1.5, YOLOv8 1.1; el clásico tarda 1.1 min en total. YOLOv9 dio primero pérdida infinita por una división por cero de su repo; se parchó (`results/e2/prueba_humo/repos_oficiales/README.md`, cambio 5) y el piloto se repitió. Suma: 44 min por época en la 3060 y ≈ 96 min en la 3050.

| E | Búsqueda (3060) | Una semilla (3060) | Búsqueda + 5 semillas (3060) | Una semilla (3050) |
|---|---|---|---|---|
| 8 | 7.1 h | 6.3 h | 38.3 h | 13.6 h |
| 7 | 7.1 h | 5.5 h | 34.6 h | 12.0 h |
| **6** | **4.8 h** | **4.8 h** | **28.7 h** | **10.4 h** |
| 5 | 4.8 h | 4.0 h | 25.0 h | 8.8 h |
| 4 | 4.8 h | 3.3 h | 21.3 h | 7.2 h |

Arrancando el 2026-11-16 hacia las 24:00, E = 7 termina hacia las 10:30 del 2026-11-18 en la 3060 sola, y la semilla 46 en la 3050 no cabe en su ventana de 10 h. **E = 6** termina hacia las 04:45 del 2026-11-18. Por eso E = 6.

**Suficiencia de E = 6 (observado en el E1, mismo lote efectivo 2).** La Mask R-CNN del E1 (12 épocas) tuvo AP de máscara en val de 0.278 / 0.264 / 0.281 en la época 6, contra un máximo de 0.294 / 0.284 / 0.289 (épocas 9, 8 y 5; semillas 42–44). SOLOv2 tuvo su mejor época en la 6, 7 y 6. Los modelos que convergen más lento (en particular Mask2Former) pueden quedar subentrenados; se declara con la regla de parada de arriba.

## 5. Preguntas de investigación y pruebas

| RQ | Tipo | Hipótesis | Prueba |
|---|---|---|---|
| RQ1. ¿Algún modelo del Top 10 supera al mejor baseline fuerte en F1 de instancia por grupo, por al menos δ? | Confirmatoria, m = 10 | H0: la mediana de la diferencia por grupo es ≤ 0. H1: es > 0. | Wilcoxon pareado de rangos con signo sobre los 14 grupos (F1 por grupo, promedio de las 5 semillas). Holm sobre los 10 contrastes. IC 95 % de la diferencia por bootstrap de grupos (10 000 réplicas). |
| RQ2. ¿La familia de dos etapas (R) supera a la de prototipos (P) en F1 por grupo? | Confirmatoria, m = 1, familia aparte | H0: la diferencia entre la media de los 4 modelos R y la de los 4 modelos P es 0. H1: es distinta de 0. | Wilcoxon pareado sobre los 14 grupos, bilateral, α = 0.05. Motivo: en la prueba de humo, P y K quedaron en 0.70–0.80 de AP50 de máscara con caja ≈ 1.0, y R en ≈ 1.0. |
| RQ3. ¿La ventaja cambia con el tamaño del tornillo? | Exploratoria | — | F1 por tercil de área. Benjamini–Hochberg. Se etiqueta como exploratorio. |
| RQ4. ¿Algún modelo sobreajusta más allá de δg? | Ruta §7 (bondad de ajuste), no confirmatoria | — | Brecha F1 train − val con IC bootstrap por grupo. |

**Mejor baseline fuerte (comparador de RQ1):** el modelo base del E1 o el baseline no profundo con mayor F1 en **val**. Se elige antes de mirar test.

**Supuestos y elección de prueba:**
- Wilcoxon es la prueba principal, porque n = 14 grupos y 7 grupos tienen una sola imagen.
- El t pareado se reporta como sensibilidad. Si Shapiro–Wilk rechaza la normalidad de las diferencias, manda Wilcoxon (en el E1 ya la rechazó, p = 0.009).
- Con 14 grupos, el p mínimo de Wilcoxon bilateral es 2/2^14 = 0.00012. Es menor que el umbral más exigente de Holm, 0.05/10 = 0.005, así que la prueba puede rechazar.
- Análisis global: Friedman con los grupos como bloques, post-hoc de Nemenyi y diagrama de diferencia crítica.

**Regla de decisión por contraste:**
- *Significativa y relevante*: p de Holm < 0.05, diferencia ≥ δ y límite inferior del IC > 0.
- *Significativa sin relevancia*: p de Holm < 0.05 pero diferencia < δ.
- *Equivalente*: IC 95 % dentro de ±δ.
- *No concluyente*: cualquier otro caso. Nunca se describe como "tendencia".

## 6. Número de ejecuciones, potencia y exclusiones

- **R = 5 semillas por modelo (42–46)**, el mínimo del PDF.
- **Potencia.** Se usaron como piloto las 15 corridas del E1 (resultados de test, solo para dimensionar; se declara como limitación). Para el par con la diferencia más pequeña (Mask R-CNN − YOLOv8):
  - Desviación por grupo × modelo σ̂ = 0.098; por semilla σ̂ = 0.087.
  - Desviación de la diferencia con R = 5: √(0.098² + 0.087²/5) = 0.105.
  - Diferencia mínima detectable con m = 10, potencia 0.80 y 14 grupos: **0.119**. Es menor que δ = 0.20, así que la potencia para detectar δ es > 0.99.
  - Para pares tan distintos como Mask R-CNN − SOLOv2 (σ̂ = 0.37), la diferencia mínima detectable sube a ≈ 0.42.
- **Por qué no más semillas:** pasar de 5 a 10 baja la diferencia mínima detectable solo de 0.119 a 0.115. El piso lo pone la variación entre escenas.
- **Exclusiones:** solo por fallo técnico (pérdida no finita, caída, sin memoria; ultralytics además falla si reduce el lote por sí solo). La corrida se repite una vez con la misma semilla; si vuelve a fallar, se reporta como fallo. Nunca se excluye por desempeño. Toda corrida descartada se lista.

## 7. Lo que no aplica o queda como exploratorio

- **Búsqueda de arquitectura (§14): no aplica.** Los 10 modelos tienen la arquitectura definida en su paper (columna F2 de `literatura/e2/lista_larga_puntaje_E2.xlsx`).
- **Exploratorios declarados**, que no se usan para elegir modelo:
  - RQ3;
  - todos los pares entre modelos, con Benjamini–Hochberg;
  - grupo 9;
  - calibración, robustez e interpretabilidad;
  - latencia;
  - con y sin preentrenamiento, solo en el mejor modelo.
- Se registra el número total de modelos, configuraciones y métricas examinados (PDF §9.3).
- **Paper guía y su limitación (PDF §5):** Kalitsios et al. 2022 (artículo original de WDSD; DOI 10.1109/ICRCV55858.2022.9953178). Es el único trabajo con el mismo dataset. Reporta con Mask R-CNN ResNet-101 un AP de 0.586 y un AP50 de 0.849, con la partición oficial (765/90/90).
  - **Limitación declarada:** esa partición es aleatoria por imagen y el 48.9 % de las imágenes de test tiene un casi gemelo en train (EDA del E1). Su AP puede estar inflado por esa fuga. Por eso el E2 usa la partición agrupada y no compara sus números con los del paper.
  - **Verificación:** exploratoria. Si sobra tiempo después de las corridas finales: ≥ 3 modelos del Top 10 entrenados con la partición oficial, comparando el AP de test en las imágenes con y sin casi gemelo (bootstrap por grupo). No se usa para elegir modelo.
