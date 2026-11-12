# Segmentación de instancias de tornillos en WDSD

**Entregable 1 del proyecto de Deep Learning**  
Ivan Berrio, Jose David Castillo y Jaime Besada

## Propósito

En el desmontaje automatizado de residuos de aparatos eléctricos y electrónicos (*Waste Electrical and Electronic Equipment*, WEEE), un robot debe localizar cada tornillo antes de retirarlo. Este proyecto estudia el problema como **segmentación de instancias**: para cada imagen se obtiene una máscara por tornillo, de modo que se distingan tanto su ubicación como su forma visible.

La pregunta que guía el trabajo es qué modelo localiza mejor los tornillos en escenas no vistas. Para responderla, se analiza el WEEE Disassembly Screw Dataset (WDSD), se revisa la literatura y se comparan tres modelos base de segmentación de instancias.

```{figure} notebooks/segmentacion-aplicacion-robotica.png
:alt: Diagrama de una imagen de entrada, una máscara de color por tornillo y un brazo robótico con su efector final.
:width: 100%
:name: esquema-segmentacion-robotica

Esquema conceptual de la segmentación de instancias y su posible aplicación robótica. Cada máscara identifica un tornillo y delimita su forma visible para apoyar su localización antes de una posible acción del efector final.
```

## Resultados principales

La métrica primaria es el F1 de instancia por imagen. Se emparejan las máscaras predichas y reales con el algoritmo húngaro; una pareja cuenta como acierto cuando su IoU de máscara es al menos 0,5. Los valores son la media y la desviación estándar entre cinco semillas:

| Modelo | F1 de instancia | PQ | AP50 de máscara |
|---|---:|---:|---:|
| Mask R-CNN | 0,447 ± 0,017 | 0,313 ± 0,011 | 0,587 ± 0,010 |
| YOLOv8-seg | 0,521 ± 0,061 | 0,365 ± 0,040 | 0,637 ± 0,019 |
| SOLOv2 | 0,297 ± 0,027 | 0,210 ± 0,017 | 0,285 ± 0,017 |

YOLOv8-seg alcanza la media más alta por imagen. En la comparación estadística por grupos de escena, SOLOv2 queda por debajo de los otros dos; la diferencia entre Mask R-CNN y YOLOv8-seg no es concluyente con los 14 grupos de prueba disponibles. Por ello, los resultados se presentan como una línea base y no como una conclusión de superioridad definitiva.

## Cómo recorrer el libro

### Datos

- {doc}`notebooks/01_eda_wdsd`: inventario, integridad y características generales del dataset. Este análisis considera la segmentación semántica y las propiedades que son comunes a ambos planteamientos.
- {doc}`notebooks/02_eda_instancias`: estructura, geometría y separación de cada tornillo como instancia; de aquí salen las decisiones específicas para el problema de manipulación.

### Modelos y resultados

- {doc}`notebooks/03_modelos_base_instancias`: revisión de literatura, protocolo de entrenamiento, métricas, resultados y análisis estadístico.
- {doc}`cumplimiento`: correspondencia de cada requisito con la evidencia y los pendientes que siguen abiertos.

El [notebook integrado del entregable](https://github.com/BESADAJ/wdsd-segmentacion-tornillos-entregable-1/blob/main/notebooks/entrega_E1_instancias.ipynb) reúne el mismo recorrido en un único documento ejecutado. El [repositorio de origen](https://github.com/BESADAJ/wdsd-segmentacion-tornillos-entregable-1) contiene además los scripts, las tablas y las salidas guardadas.

## Alcance y límites de la evidencia

El conjunto contiene 945 imágenes de 1280 × 720 y 4.398 instancias anotadas. La partición mantiene juntas las imágenes de una misma escena: 744 para entrenamiento, 93 para validación y 108 para prueba. En la prueba hay 14 grupos, por lo que la incertidumbre estadística se calcula remuestreando escenas y no imágenes aisladas.

El margen práctico es δ = 0,20 puntos de F1 y se propone para juzgar modelos avanzados frente a la línea base. Se eligió después de obtener los resultados de estos tres modelos, así que no fue un criterio previo para decidir entre ellos. El informe contrasta su motivación con un proceso físico de atornillado, pero la medida física reportada en ese trabajo no se convierte directamente en puntos de F1; δ se interpreta dentro de este benchmark y no como una tolerancia industrial validada.

La guía del curso presenta FCN, codificador–decodificador y U-Net reducida como ejemplos generales de segmentación. Aquí se eligieron modelos de segmentación de instancias porque la salida distingue cada tornillo; por eso la terna no coincide literalmente con esos ejemplos generales.

La comparación no incluye búsqueda equivalente de hiperparámetros para las tres familias ni medición separada del tiempo de inferencia. Tampoco se afirma que el éxito de segmentación garantice que un robot pueda retirar el tornillo: eso requiere validación física. La página pública del dataset no especifica condiciones de uso y redistribución, y queda pendiente auditar tornillos visibles sin anotación.
