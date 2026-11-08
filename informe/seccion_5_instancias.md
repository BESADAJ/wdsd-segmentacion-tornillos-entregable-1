# 5. Línea base estadística

## 5.1 Protocolo experimental

Se entrenaron los tres modelos base (Mask R-CNN, YOLOv8-seg y SOLOv2) con cinco semillas cada uno (42 a 46): 15 corridas en total. Todas las corridas comparten el mismo protocolo:

- **Partición.** La partición agrupada congelada del EDA (`results/eda/particion.csv`): 744 imágenes de entrenamiento, 93 de validación y 108 de prueba. En prueba hay 14 grupos (sesiones o escenas de captura). Ninguna sesión aparece en más de un conjunto.
- **Entrada.** Imagen completa de 1280×720, sin reescalar ni recortar, porque el diámetro mediano de un tornillo es de 11.7 px (sección 4). El único aumento de datos es el volteo horizontal.
- **Inicialización.** Pesos preentrenados en COCO para los tres modelos.
- **Selección del modelo.** Se guarda la mejor época según el conjunto de validación: AP de máscara en Mask R-CNN y SOLOv2, y el *fitness* de ultralytics en YOLOv8-seg. El conjunto de prueba se evalúa una sola vez por corrida.
- **Umbral de confianza.** Las métricas por instancia (F1 y PQ) necesitan un umbral sobre el puntaje de cada detección. Se elige en validación, como el que maximiza el F1 medio por imagen, y se aplica sin cambios en prueba. El AP no usa umbral.

Tabla 5.1. Configuración por modelo.

| Modelo | Tipo | Parámetros | Épocas | Lote | Optimizador | Marco |
|---|---|---|---|---|---|---|
| Mask R-CNN (ResNet-50-FPN) | Dos etapas, con cajas | 43 922 395 | 12 | 2 | SGD, momento 0.9, calentamiento + coseno, lr 0.0025 × lote | torchvision 0.26 |
| YOLOv8n-seg | Una etapa, con cajas | 3 263 811 | 40 | 4 | Valores por defecto de ultralytics | ultralytics 8.4.207 |
| SOLOv2 (ResNet-50-FPN) | Una etapa, sin cajas | 46 228 801 | 12 | 2 | SGD, lr 0.01 × lote/16 (receta oficial `solov2_r50_fpn_1x_coco`) | MMDetection 3.3 |

Mask R-CNN usa anclas de 8 a 128 px en lugar de las de 32 a 512 px que trae torchvision por defecto, porque el 97 % de las cajas mide menos de 32 px (sección 4).

**Hardware y reparto.** Las corridas se repartieron entre dos máquinas por semilla, no por modelo:
- semillas 42, 43 y 44 en una RTX 3060 Laptop de 6 GB;
- semillas 45 y 46 en una RTX 3050 Laptop de 6 GB.

Las dos usaron la misma configuración y las mismas versiones de los paquetes (`requirements_wdsd_dl_congelado.txt`). Así, cualquier efecto de la máquina cae por igual sobre los tres modelos.

## 5.2 Métricas

- **Primaria, declarada a priori: F1 de instancia por imagen.** En cada imagen se emparejan predicciones y tornillos reales con el algoritmo húngaro. Una pareja cuenta como acierto (TP) si su IoU de máscara es de al menos 0.5. Con eso, F1 = 2·TP / (2·TP + FP + FN). FP son las predicciones sin pareja y FN los tornillos sin detectar.
- **Secundarias:**
  - PQ (*panoptic quality*) = SQ × F1, donde SQ es el IoU medio de los aciertos;
  - AP50 de máscara.
- **Informativa:** AP de máscara promediado en IoU de 0.5 a 0.95, el estándar de COCO.
- **Umbral práctico:** δ = 0.03 en F1, declarado a priori en el EDA.

## 5.3 Resultados

Tabla 5.2. Desempeño en prueba: media ± desviación estándar sobre 5 semillas, con intervalo de confianza del 95 % (t de Student, entre corchetes).

| Modelo | F1 por imagen | PQ por imagen | AP50 | AP | AP75 |
|---|---|---|---|---|---|
| Mask R-CNN | 0.447 ± 0.017 [0.425, 0.468] | 0.313 ± 0.011 [0.300, 0.327] | 0.587 ± 0.010 | 0.217 ± 0.004 [0.212, 0.222] | 0.085 ± 0.011 |
| YOLOv8-seg | 0.521 ± 0.061 [0.445, 0.597] | 0.365 ± 0.040 [0.316, 0.415] | 0.637 ± 0.019 | 0.234 ± 0.008 [0.224, 0.245] | 0.094 ± 0.018 |
| SOLOv2 | 0.297 ± 0.027 [0.263, 0.330] | 0.210 ± 0.017 [0.189, 0.231] | 0.285 ± 0.017 | 0.113 ± 0.005 [0.106, 0.119] | 0.062 ± 0.011 |

Tabla 5.3. Costo computacional por corrida (entrenamiento + evaluación).

| Modelo | Minutos, RTX 3060 (3 semillas) | Minutos, RTX 3050 (2 semillas) | Memoria GPU pico (GB) |
|---|---|---|---|
| Mask R-CNN | 33.5 | 73.5 | 2.7 |
| YOLOv8-seg | 21.0 | 62.4 | 2.6 |
| SOLOv2 | 31.7 | 55.9 | ~2.6* |

\* Para SOLOv2 se toma el pico del registro de MMDetection durante el entrenamiento; el contador de PyTorch del script se reinicia dentro de MMDetection y da un valor menor que no corresponde al entrenamiento.

Tabla 5.4. Conteos agregados en prueba (5 semillas × 108 imágenes; 1945 tornillos reales en total).

| Modelo | Aciertos (TP) | Falsos positivos | Tornillos no detectados | Precisión | Recall |
|---|---|---|---|---|---|
| Mask R-CNN | 1132 | 742 | 813 | 0.60 | 0.58 |
| YOLOv8-seg | 1220 | 706 | 725 | 0.63 | 0.63 |
| SOLOv2 | 595 | 410 | 1350 | 0.59 | 0.31 |

## 5.4 Comparación estadística

**Unidad de análisis.** La unidad de independencia es el grupo, no la imagen, porque las imágenes de una misma sesión están correlacionadas. Para cada modelo y grupo de prueba se promedia el F1 de las imágenes del grupo y luego el de las semillas. Resulta una tabla de 14 grupos × 3 modelos.

**Pruebas.**
1. **Friedman** sobre los tres modelos, con los grupos como bloques. Prueba si al menos un modelo difiere de los otros.
2. **Wilcoxon pareado por grupo** para cada par de modelos, con corrección de Holm por los 3 contrastes.
3. **Intervalo bootstrap del 95 %** de la diferencia media por par, remuestreando grupos (10 000 réplicas).
4. **Efecto mínimo detectable (MDE)** de cada par, con el alfa de Holm y potencia 0.80. Si el MDE es mayor que δ, la prueba no tiene potencia para detectar una diferencia práctica, y un resultado no significativo se reporta como no concluyente, no como empate.

Resultados:
- Friedman: χ² = 16.62, p = 0.0002. Al menos un modelo difiere de los otros.

Tabla 5.5. Contrastes por pares (unidad = grupo, n = 14).

| Par | Diferencia media en F1 | IC95 bootstrap | p (Wilcoxon) | p (Holm) | Grupos donde gana el primero | MDE | ¿Supera δ? |
|---|---|---|---|---|---|---|---|
| Mask R-CNN − SOLOv2 | +0.457 | [0.268, 0.639] | 0.0002 | 0.0007 | 13/14 | 0.356 | Sí |
| Mask R-CNN − YOLOv8-seg | −0.009 | [−0.066, 0.039] | 0.553 | 0.553 | 9/14 | 0.102 | No (no concluyente) |
| SOLOv2 − YOLOv8-seg | −0.465 | [−0.641, −0.285] | 0.0006 | 0.0012 | 1/14 | 0.337 | Sí |

Con PQ como métrica secundaria el patrón es el mismo: Friedman p = 0.0002; Mask R-CNN − YOLOv8-seg = +0.011 (IC95 [−0.037, 0.053], p de Holm = 0.27, MDE 0.088); SOLOv2 pierde contra los dos (p de Holm ≤ 0.0007).

**Lectura.**
- **SOLOv2 queda por debajo de los otros dos** con una diferencia grande (≈0.46 en F1 por grupo), muy por encima de δ = 0.03 y significativa tras Holm. Pierde en 13 de los 14 grupos frente a cada uno de los otros modelos. La tabla 5.4 muestra de dónde viene: su precisión es similar a la de los otros (0.59 frente a 0.60–0.63), pero su recall es la mitad (0.31 frente a 0.58–0.63). Deja de detectar la mitad de los tornillos. Una explicación plausible, no verificada con un experimento, es su forma de asignar objetos: SOLOv2 decide la presencia de una instancia en una cuadrícula de 40 × 40 celdas, que a 1280 px son celdas de unos 32 px, mientras que el tornillo mediano mide 12 px.
- **Mask R-CNN y YOLOv8-seg no se distinguen con estos datos.** La diferencia por grupo es de −0.009 (IC95 [−0.066, 0.039]; p de Holm = 0.55). El efecto mínimo detectable con 14 grupos es de 0.102, mayor que δ = 0.03, así que el resultado es **no concluyente**: no se puede afirmar que sean equivalentes, solo que la prueba no tiene potencia para separarlos.
- **La media por imagen y la media por grupo cuentan historias distintas.** Por imagen, YOLOv8-seg supera a Mask R-CNN (0.52 frente a 0.45), pero buena parte de esa ventaja viene del grupo 9, que tiene 28 de las 108 imágenes de prueba: allí Mask R-CNN casi no acierta (F1 0.02) y YOLOv8-seg llega a 0.22. Al usar el grupo como unidad, ese grupo pesa lo mismo que los demás y la ventaja desaparece. Por eso la conclusión se apoya en el análisis por grupo.
- **El reparto entre máquinas no cambia el patrón.** Las semillas de la RTX 3050 (45 y 46) quedan en el mismo rango que las de la RTX 3060 en F1 por imagen: Mask R-CNN 0.450 y 0.461 frente a 0.418–0.458; YOLOv8-seg 0.568 y 0.504 frente a 0.429–0.584; SOLOv2 0.305 y 0.287 frente a 0.256–0.326. Lo que cambia es el tiempo de cómputo.

## 5.5 Limitaciones

- **Pocos grupos en prueba.** Hay 14, y 8 de ellos tienen una o dos imágenes (7 con una sola). Eso limita la potencia: con Wilcoxon exacto y 14 pares, el p más pequeño posible es del orden de 10⁻⁴, pero detectar diferencias moderadas exige que un modelo gane en casi todos los grupos.
- **Marcos distintos.** SOLOv2 corre en MMDetection, con torch 2.1.2, porque mmcv 2.1 solo tiene versiones precompiladas para Windows con torch 2.1. Mask R-CNN y YOLOv8-seg usan torch 2.11. La evaluación es idéntica para los tres (el mismo código de métricas sobre las predicciones exportadas en formato COCO), pero las rutinas de entrenamiento no son idénticas.
- **Hiperparámetros por defecto.** Cada modelo usa la receta de su implementación de referencia, solo ajustada al tamaño de los tornillos (anclas, resolución). No hubo búsqueda de hiperparámetros, así que la comparación es entre recetas estándar, no entre el máximo alcanzable por cada arquitectura.
- **Presupuesto fijo de épocas.** 12/40/12 épocas por limitaciones de tiempo y GPU (6 GB). Un modelo que converja más lento queda en desventaja.
- **No determinismo de GPU.** Con la misma semilla, cuDNN no garantiza resultados idénticos. La variación entre semillas incluye ese ruido.
