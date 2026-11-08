# Lectura crítica de los 9 estudios de segmentación de instancias

Insumo para las secciones 3.2 y 3.4 del Entregable 1. Datos en `tabla_sintesis_instancias.csv`, extraídos del texto completo de cada PDF (carpeta `revision de literatura/instance segmentation`). Búsqueda C4 del 2026-11-07; diagrama en `prisma_instancias.png`.

Los 9 estudios son 7 de la búsqueda (I1–I7) y 2 de otras fuentes: Kalitsios 2022 (artículo original de WDSD) y Chao 2025 (único trabajo posterior que usa WDSD).

## Cuántos estudios usan cada modelo como comparación

| Modelo | Estudios (de 9) | Cuáles |
|---|---|---|
| Mask R-CNN | 6 | I1, I2, I3, I5, I6, Kalitsios 2022 |
| YOLO-seg (v5/v8/v11) | 5 | I4, I5, I6, I7, Chao 2025 |
| YOLACT | 3 | I1, I5, I6 |
| SOLO / SOLOv2 | 2 | I2 (SOLO), I5 (SOLOv2) |
| BlendMask | 2 | I2, I5 |

En los 219 resúmenes únicos de la búsqueda los conteos son: YOLO (cualquiera) 64, Mask R-CNN 59, YOLO-seg 25, SOLO/SOLOv2 10 y YOLACT 7.

**Por qué SOLOv2 y no YOLACT (limitación que se declara).** En texto completo YOLACT aparece en 3 estudios y SOLO/SOLOv2 en 2; en los resúmenes el orden se invierte (10 contra 7). Se eligió SOLOv2 por dos razones: es más frecuente en el universo de resúmenes y completa la no redundancia de la terna, porque es el único de los candidatos que segmenta sin cajas. Con solo 9 estudios en texto completo, una diferencia de 3 contra 2 no permite ordenar con confianza.

## La terna cubre tres familias distintas

| Modelo | Familia | Cómo obtiene cada instancia |
|---|---|---|
| Mask R-CNN | Dos etapas, con cajas | Propone regiones, recorta sus características y predice una máscara por región |
| YOLOv8-seg | Una etapa, con cajas | Predice cajas y coeficientes en una pasada; la máscara es una combinación de prototipos compartidos |
| SOLOv2 | Una etapa, sin cajas | Divide la imagen en una rejilla; cada celda que contiene el centro de un objeto genera directamente su máscara |

## Ningún orden es estable entre los modelos

- **AF-YOLO, I5** (alambres de seguridad, objetos de menos del 1 % de la imagen, como nuestros tornillos), mAP de máscara: Mask R-CNN 83.2 < YOLACT 84.7 < SOLOv2 86.1 < YOLOv8-seg 87.3. Es la única comparación que incluye los tres modelos de nuestra terna.
- **SPGNet, I6** (piezas de vehículos fuera de uso), mAP@0.5 de máscara: Mask R-CNN 78.3, YOLOv5s-seg 90.1, YOLACT 90.5. Mask R-CNN queda 12 puntos por debajo.
- **I2** (conteo de sujetadores densos): SOLO tiene menos error de conteo que Mask R-CNN (2.21 % contra 8.84 % con umbral 0.3), pero mucho peor distancia al centro de gravedad (47.13 contra 4.38). El orden depende de la métrica.
- **LPHD, I3** (tornillos hexagonales): Mask R-CNN tiene mayor IoU que la propuesta (92.9 % contra 91.4 %), pero es 2.5 veces más lento.
- Lectura: el orden cambia con el dataset y con la métrica, y ningún estudio da la dispersión entre corridas que permitiría saber si las diferencias superan el ruido.

## Calidad metodológica

| Criterio | Resultado en los 9 |
|---|---|
| Más de una corrida o semilla | 0 de 9 |
| Reporta dispersión (± DE o IC) | 0 de 9 |
| Prueba estadística | 0 de 9. I4 e I5 dicen "significantly" sin prueba |
| Partición agrupada por sesión u objeto | 1 de 9 (I7, agrupa por zapato) |
| Aumento de datos antes de partir (fuga) | 3 de 9 (I1, I4, Chao 2025) |
| Sin test separado | I2 (366/47 sin test) e I6 (7:3 train/val) |
| Comparación incompleta | I1 compara con Mask R-CNN solo por curvas de entrenamiento, sin métricas de test |
| Partición con fuga conocida en WDSD | Kalitsios 2022 usa la oficial 765/90/90: según nuestro EDA, 48.9 % de test tiene una casi-gemela en train |

## Antecedentes sobre WDSD

- **Kalitsios et al. 2022** (DOI 10.1109/ICRCV55858.2022.9953178): solo Mask R-CNN (ResNet-101, preentrenada en COCO), AP0.50:0.95 = 0.586 y AP0.50 = 0.849, una corrida, partición oficial.
- **Chao et al. 2025** (DOI 10.1080/27525783.2025.2551315): YOLOv8-SM-Seg para tornillos y agujeros, con 387 imágenes de WDSD más 301 propias, aumentadas a 903 antes de partir al azar; una corrida, mAP 96.4 %. Otra tarea y otro conjunto, así que no es comparable con nuestro resultado.

## Vacío propuesto (formulación verificable)

Ningún estudio incluido compara Mask R-CNN, YOLOv8-seg y SOLOv2 en segmentación de instancias de tornillos con una partición agrupada por sesión de captura, varias semillas e inferencia estadística. Los dos trabajos sobre WDSD usan un solo modelo cada uno, una sola corrida y particiones con riesgo de fuga.

## Datos útiles para la sección 3.3

- I5 da la referencia más cercana a nuestro caso (objetos pequeños): los tres modelos de la terna quedan dentro de 4.1 puntos de mAP entre sí.
- I2 respalda una métrica secundaria de localización por centroide, útil para la idea del equipo sobre la precisión de la posición del tornillo.
- I4 reporta YOLOv8n-seg con 3.26 M parámetros y YOLOv8s-seg con 11.78 M. Nuestro conteo con ultralytics 8.4.207 da 3.26 M (n) y 11.79 M (s) con una sola clase, así que coincide (diferencia ≤ 0.01 M).

## Pendiente

- Sin texto completo (no recuperados en el PRISMA): Bolt Sorting (10.1109/case59546.2024.10711806), remaches de tren, componentes de vía férrea y sujetadores de vía.
