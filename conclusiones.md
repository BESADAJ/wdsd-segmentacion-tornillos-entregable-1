# 6. Conclusiones

1. **El problema está bien planteado como segmentación de instancias.** Los tornillos son pequeños (diámetro mediano de 10.6 px) y no se tocan entre sí, lo que obliga a trabajar a resolución completa y a ajustar las anclas (EDA 2).
2. **La literatura no ofrece un orden estable entre los modelos ni comparaciones con rigor estadístico.** Ninguno de los 9 estudios usa más de una corrida ni pruebas estadísticas, y solo uno usa partición agrupada (capítulo 3). Este trabajo compara los tres modelos más usados como referencia con partición agrupada, 5 semillas e inferencia por grupo.
3. **Línea base (capítulo 5):**
   - YOLOv8-seg y Mask R-CNN quedan en el mismo nivel (F1 por imagen de 0.52 y 0.45; diferencia por grupo de −0.009, no concluyente con 14 grupos).
   - SOLOv2 queda claramente por debajo (F1 0.30), porque deja sin detectar cerca de dos tercios de los tornillos.
4. **Baseline y qué sigue (E2).** El baseline es YOLOv8-seg (F1 por imagen de 0.521, IC95 [0.445, 0.597]), al nivel de Mask R-CNN por grupo. Un modelo avanzado debe alcanzar al menos 0.721 de F1 por imagen para superar el baseline por δ = 0.20. El MDE actual por grupos es ≈0.10; en el E2 debe recalcularse con el diseño final y sus comparaciones. Ampliar el número de escenas también permitirá evaluar mejor la generalización. Los grupos difíciles (por ejemplo, el grupo 9) son el principal margen de mejora.
