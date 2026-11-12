# Correspondencia con el enunciado

Esta tabla permite ubicar la evidencia del Entregable 1 y distinguir lo que está documentado de lo que todavía no se midió o verificó.

| Requisito | Evidencia en el proyecto | Estado | Observación |
|---|---|---|---|
| Formular la tarea de aprendizaje | {doc}`notebooks/02_eda_instancias`, secciones 2.1 y 2.13 | Atendido | Entrada: imagen RGB de 1280 × 720; salida: una máscara por tornillo. Se define la pregunta de investigación y se declara el grupo de escena como unidad de independencia. |
| Analizar el dataset antes de modelar | {doc}`notebooks/01_eda_wdsd` y {doc}`notebooks/02_eda_instancias` | Atendido | El primer EDA describe el dataset y la segmentación semántica; el segundo analiza las instancias y justifica decisiones para localizar tornillos pequeños. |
| Controlar fuga entre particiones | `results/eda/particion.csv`; sección 2.10 del EDA de instancias | Atendido | La partición mantiene juntas las imágenes de una misma escena. La prueba tiene 108 imágenes de 14 grupos. |
| Revisar anotaciones y trazabilidad | Secciones 2.2–2.3 del EDA de instancias y EDA general | Parcial | Se estudian estructura, polígonos y consistencia entre formatos. Falta auditar una muestra de tornillos visibles sin etiqueta. |
| Documentar condiciones de uso del dataset | README y sección de datos | Pendiente | Las fuentes públicas consultadas no especifican una licencia ni condiciones de redistribución. La disponibilidad de descarga no resuelve este punto. |
| Revisar literatura y escoger modelos | {doc}`notebooks/03_modelos_base_instancias`, secciones 3.1–3.4 | Atendido con alcance delimitado | Se aplicó un protocolo PRISMA y se sintetizaron nueve estudios. Se comparan Mask R-CNN, YOLOv8-seg y SOLOv2, modelos que producen máscaras por instancia. La guía general del curso también menciona FCN, codificador–decodificador y U-Net reducida; esa terna no se entrenó aquí. |
| Implementar y repetir los modelos base | Secciones 4.1–4.4; `results/instancias/corridas/` | Atendido | Hay cinco semillas por modelo y 15 corridas en total, con partición, hardware, configuraciones y salidas descritos. |
| Hacer comparables las recetas de entrenamiento | Secciones 4.1 y 5.5 | Parcial | Se mantuvieron particiones y evaluación comunes, pero las recetas, aumentos y presupuestos de épocas difieren. No hubo una búsqueda comparable de hiperparámetros. |
| Definir y reportar métricas de instancia | Secciones 2.13 y 5.1–5.3 | Atendido | Métrica primaria: F1 por imagen, con emparejamiento húngaro y umbral IoU de 0,5. Se reportan PQ y AP50; AP de 0,5 a 0,95 es informativa. |
| Justificar el margen práctico δ | Secciones 2.13, 4.5 y 5.4–5.5 | Parcial | Se usa δ = 0,20 puntos de F1 para valorar modelos futuros. Se eligió después de las corridas base; el artículo físico sobre atornillado no permite convertir milímetros de posicionamiento a puntos de F1. |
| Reportar métricas de píxel del benchmark | Sección 5.1 | Pendiente | Los resultados de los modelos se presentan con métricas de instancia y de máscara. No se reportan Dice, IoU ni exactitud de píxel por imagen para las 15 corridas. |
| Incluir ejemplos cualitativos y costo de inferencia | Sección 5.5 | Pendiente | No se presentan predicciones frente a la verdad de referencia ni tiempo de inferencia aislado. Las predicciones COCO están guardadas, pero este análisis no se realizó. |
| Presentar inferencia estadística y límites | Secciones 4.5 y 5.3–5.5 | Atendido con límites | Se comparan grupos de escena con pruebas pareadas y corrección de Holm; se informa un efecto mínimo detectable. El dimensionamiento estadístico se hizo después de las corridas y la prueba solo contiene 14 grupos. |
| Entregar un informe reproducible y legible | Este libro, los notebooks, `src/`, scripts, `results/` y archivos `requirements*.txt` | Parcial | El libro se construye sin descargar datos y conserva resultados ejecutados. El dataset no se incluye en el repositorio y requiere descarga externa; sus términos de uso aún no están aclarados. |

## Lectura del estado

El núcleo del trabajo —EDA, revisión de literatura, 15 corridas y comparación estadística— está documentado. Las brechas abiertas se concentran en la licencia y auditoría de anotaciones, la diferencia entre los modelos de instancia entrenados y los ejemplos generales de la guía, la falta de métricas de píxel y ejemplos cualitativos, y el tiempo de inferencia. No se presentan esos puntos como si ya estuvieran resueltos.
