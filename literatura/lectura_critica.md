# Lectura crítica de los 12 estudios de la tabla de síntesis

Insumo para las secciones 3.2 y 3.4 del Entregable 1. Datos en `tabla_sintesis.csv`, extraídos del texto completo de cada PDF (carpeta `revision de literatura/semantic segmentation`). Fecha: 2026-11-07.

## Cuántos estudios usan cada modelo base como comparación

| Modelo | Estudios | Cuáles |
|---|---|---|
| U-Net (o variante directa) | 9 | Feng 2026, Xiang 2026, Alvares 2026, Vihuro 2026, Jiang 2023, Mohamed 2024, Huang 2023, Wang 2026, Niu 2023 (ResUnet) |
| FCN | 5 | Feng 2026, Alvares 2026, Jiang 2023, Jiang 2021, Wang 2026 |
| SegNet | 5 | Feng 2026, Vihuro 2026, Mohamed 2024, Huang 2023, Jiang 2021 |

DeepLab aparece en 6, pero siempre como modelo más pesado (54–59 M parámetros en Xiang 2026 y Jiang 2023).

## Ningún orden es estable entre los tres modelos base

- FCN supera a U-Net en Feng 2026 (256×256), Alvares 2026 y Jiang 2023 (por 0.06 puntos). U-Net supera a FCN por 26 puntos de mIoU en Wang 2026.
- SegNet supera a U-Net en Huang 2023 y en Feng 2026 (1024×1024). U-Net supera a SegNet en Vihuro 2026, Mohamed 2024 y Feng 2026 (256×256).
- Lectura: el orden depende del dataset y del protocolo. Ningún estudio aporta la dispersión entre corridas que permitiría saber si esas diferencias superan el ruido.

## Calidad metodológica (criterios del punto 3.1)

| Criterio | Resultado en los 12 |
|---|---|
| Más de una corrida o semilla | 1 de 12 (Xiang 2026: 5 semillas) |
| Reporta dispersión (± DE o IC) | 1 de 12, y solo en la ablación (Xiang 2026) |
| Prueba estadística | 0 de 12. Feng 2026 y Alvares 2026 dicen "significantly" sin prueba |
| Partición agrupada por sesión, lote u objeto | 0 de 12 |
| Riesgo alto de fuga | Jiang 2021 y Chao 2025 parten después del aumento; Kalitsios 2022 usa la partición oficial de WDSD (48.9 % de test con casi-gemela en train, según nuestro EDA) |
| Sin test separado | Alvares 2026 y Jiang 2023 reportan sobre validación |
| Inconsistencias internas | Alvares 2026 (U-Net 0.6611 en tabla vs. 0.7646 en texto); Mohamed 2024 (tabla y texto no coinciden) |
| Modelos comparados bajo el mismo protocolo | No en Alvares 2026 (U-Net tomada de otro artículo) |

## Gap propuesto (formulación verificable)

Ningún estudio incluido compara FCN, SegNet y U-Net para segmentación semántica de tornillos bajo una partición agrupada, con varias semillas e inferencia estadística. Los dos trabajos sobre WDSD (Kalitsios 2022 y Chao 2025) hacen segmentación de instancias, con una sola corrida y particiones con riesgo de fuga.

## Datos útiles para la sección 3.3

- Jiang 2023: U-Net con ancho base 16 tiene 1.94 M parámetros (igual que nuestra implementación) y rinde casi igual que la de ancho 64 (98.30 % vs 98.33 % de mIoU). Respalda usar una "U-Net reducida".
- Wang 2026: FCN-8s con VGG16 tiene 134.43 M parámetros y U-Net 31.04 M. Nuestra FCN (1.18 M) comparte el codificador con las otras dos, así que la comparación aísla el decodificador.
- Pérdidas reportadas: BCE + Dice (Wang 2026, igual que la nuestra), BCE + Jaccard (Mohamed 2024), BCE (Huang 2023), entropía cruzada (Feng 2026).
- Niu 2023 evalúa la localización por objeto a partir de la máscara (precisión y recall por tornillo). Respalda la métrica secundaria de localización.

## Pendiente

- #1 (braided composites), #3 (machine tool components) y #12 (abnormality of screws): sin acceso. Se declaran como no recuperados en el PRISMA (etapa de elegibilidad).
- DOI de Kalitsios 2022: 10.1109/ICRCV55858.2022.9953178 (aportado por Ivan).
