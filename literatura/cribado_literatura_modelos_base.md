# Cribado de literatura: 3 modelos base (WDSD, segmentación de tornillos)

Fecha: 2026-11-07. Fuente: exportes de Ivan de WoS, Scopus e IEEE Xplore (cadenas C1, C2 y C3), recibidos el 2026-11-07.
Lista completa con decisión por registro: `cribado_registros.csv` (misma carpeta).

## Cómo se hizo y qué limita el resultado

- **Cribado automático por reglas sobre título y resumen.** No es lectura humana. El PDF pide indicar quién hizo la selección y cómo se resolvieron desacuerdos, y una etapa de **elegibilidad por texto completo**. Ninguna de las dos está hecha.
- Las reglas usan palabras clave. Hay errores conocidos: un artículo médico que usa la palabra "disassembly" se coló (cirugía de cataratas), y la regla de dominio puede dejar fuera o dentro casos límite. La columna `revision_manual` del CSV está para corregirlos.
- "Arquitectura en frase comparativa" es un **indicador indirecto** de uso como línea base: la arquitectura aparece en una frase del resumen que también contiene compar*, baseline, outperform, than, versus o state-of-the-art. La tabla de frecuencia definitiva exige confirmar en el texto completo qué modelo es línea base.

## Flujo tipo PRISMA (borrador)

| Etapa | Registros |
|---|---|
| Identificados | 1203 (WoS 251, Scopus 598, IEEE Xplore 354) |
| Por cadena | C1: WoS 81, Scopus 288, IEEE 265 · C2: WoS 26, Scopus 54, IEEE 33 · C3: WoS 144, Scopus 256, IEEE 56 |
| Duplicados eliminados (DOI o título) | 300 |
| Cribados por título y resumen | 903 |
| Excluidos | 518 |
| ↳ E0 no es artículo o sin resumen | 20 |
| ↳ E1 fuera de 2018–2026 | 38 |
| ↳ E2 no usa deep learning | 77 |
| ↳ E3 no es segmentación | 130 |
| ↳ E4 dominio médico o biológico | 184 |
| ↳ E5 teledetección, agro, tráfico, documentos | 41 |
| ↳ E6 sin relación industrial o mecánica | 28 |
| Incluidos tras cribado automático | 385 |
| ↳ núcleo: tornillos, sujetadores, pernos, desmontaje | 118 |
| ↳ ampliado: segmentación industrial, defectos, piezas | 267 |
| Segunda pasada (lectura de resumen) | 206 incluidos, 179 excluidos (155 + 24 dudosos excluidos provisionalmente) |
| Elegibilidad por texto completo | **pendiente** |

## Frecuencia de arquitecturas (registros incluidos)

### Núcleo (tornillos y desmontaje, 118 registros)

| Arquitectura | Mencionada | En frase comparativa |
|---|---|---|
| YOLO (detección o YOLO-seg) | 27 | 6 |
| U-Net | 18 | 4 |
| Mask R-CNN | 15 | 4 |
| DeepLab | 7 | 2 |
| FCN | 5 | 1 |
| PSPNet | 3 | 1 |
| SegNet | 0 | 0 |

Lectura: en tornillos domina la **detección** (YOLO) y la **segmentación de instancias** (Mask R-CNN). Los estudios de **segmentación semántica** de tornillos son pocos (29 mencionan "semantic segmentation", varios sobre nubes de puntos de vías férreas). Con este núcleo solo no sale una terna sólida de modelos base de segmentación semántica.

### Ampliado (segmentación industrial y defectos, 267 registros)

| Arquitectura | Mencionada | En frase comparativa |
|---|---|---|
| U-Net | 179 | 105 |
| DeepLab (v3/v3+) | 47 | 38 |
| Transformer (SegFormer, Swin, ViT) | 27 | 18 |
| U-Net++ | 20 | 17 |
| SegNet | 19 | 13 |
| PSPNet | 15 | 13 |
| FCN | 16 | 9 |
| YOLO | 14 | 10 |
| LinkNet | 6 | 6 |

## Segunda pasada: cribado asistido por IA (lectura de cada resumen)

Método: Claude leyó título y resumen de los 385 registros que pasaron el filtro automático y decidió incluir, excluir o marcar como dudoso. Declarar en el informe como **"cribado asistido por IA con verificación humana de los casos dudosos"**. Los 24 dudosos quedan con `revision_manual = PENDIENTE: verificar` en el CSV (columnas nuevas `decision_ia`, `motivo_ia`, `categoria_ia`, `baselines_resumen_ia`).

| Etapa | Registros |
|---|---|
| Entran a la segunda pasada | 385 |
| Excluidos | 155 |
| ↳ E2 no usa deep learning | 4 |
| ↳ E3 no hace segmentación (solo detección o clasificación) | 31 |
| ↳ E4 médico o biológico | 8 |
| ↳ E5 otro dominio (teledetección, agro, documentos, humo, etc.) | 52 |
| ↳ E6 fuera de lo industrial o infraestructura | 43 |
| ↳ E7 no es imagen 2D (nube de puntos, señal, radar) | 17 |
| Dudosos: excluidos provisionalmente por información insuficiente en el resumen (revisión manual antes de la entrega) | 24 |
| **Incluidos** | **206** (núcleo tornillos/fijaciones/RAEE 53 · industrial/defectos 121 · infraestructura civil 32) |

### Arquitecturas usadas o comparadas según el resumen (206 incluidos)

Cuenta = número de estudios que la mencionan como modelo propio o comparado (una vez por estudio). 42 incluidos no nombran ninguna arquitectura en el resumen: hay que mirarlas en texto completo.

| Arquitectura | Tipo | Todos (206) | Núcleo + industrial (174) | Solo núcleo (53) |
|---|---|---|---|---|
| U-Net | semántica | 107 | 90 | 15 |
| DeepLab (v3/v3+) | semántica | 30 | 21 | 5 |
| YOLO-seg / YOLO | instancias o detección | 23 | 23 | 18 |
| Mask R-CNN | instancias | 17 | 14 | 11 |
| **FCN** | semántica | **13** | **10** | 1 |
| SegNet | semántica | 9 | 8 | 0 |
| PSPNet | semántica | 9 | 6 | 2 |
| U-Net++ | semántica (variante U-Net) | 9 | 8 | 0 |
| SegFormer | semántica (transformer) | 7 | 3 | 0 |

YOLO-seg y Mask R-CNN segmentan **instancias** (cada tornillo por separado), no la máscara binaria tornillo/fondo del proyecto. Por eso no cuentan como candidatos a modelo base, aunque dominan en el núcleo.

**Qué cambia respecto al conteo por palabras clave (observado):** con lectura de resumen, **FCN (13) queda por encima de SegNet (9)**. El conteo automático (solo frases comparativas) tenía a FCN por debajo (9 vs 13). No verifiqué la causa exacta de la diferencia. El orden de las semánticas queda: U-Net, DeepLab, FCN, y luego SegNet, PSPNet y U-Net++ empatadas.

### Casos dudosos (excluidos provisionalmente; revisión manual pendiente antes de la entrega)

Criterio: decide si el estudio hace segmentación con deep learning sobre imágenes 2D en un dominio industrial o de infraestructura. Escribe "incluir" o "excluir" en `revision_manual`.

| id | Año | Título | DOI |
|---|---|---|---|
| 10 | 2020 | A Lightweight and Efficient Network for Logistics Truck Scene Semantic Segmentation | https://doi.org/10.1109/iccc51575.2020.9345246 |
| 56 | 2025 | CSDCNet: A Semantic Segmentation Network for Tubular Structures | https://doi.org/10.1007/978-981-96-0116-5_24 |
| 152 | 2023 | Industrial smoke image segmentation based on improved attention W-Net [基于改进注意力W-Net的工业烟尘图像分割] | https://doi.org/10.13196/j.cims.2023.02.023 |
| 158 | 2024 | Integration Artificial Intelligence with Conventional X-Ray Inspection for Improved Solder Void Detection in SSDC SiC Modules | https://doi.org/10.1109/iicaiet62352.2024.10730272 |
| 160 | 2019 | Intelligent Localization of Transformer Internal Degradations Combining Deep Convolutional Neural Networks and Image Segmentation | https://doi.org/10.1109/access.2019.2916461 |
| 165 | 2025 | Introduction to U-NET Variant Algorithm Infused from Transfer Learning | https://doi.org/10.1109/icad65464.2025.11114049 |
| 166 | 2018 | Knowledge-Graph Based Multi-Target Deep-Learning Models for Train Anomaly Detection | https://doi.org/10.1109/icirt.2018.8641687 |
| 230 | 2024 | Saliency-based Explainable Artificial Intelligence (XAI) for Interpreting U-Net Semantic Segmentation of Synthetic Liquid Spray Images | https://doi.org/10.1109/icoco62848.2024.10928179 |
| 240 | 2022 | Semantic segmentation based stereo visual servoing of nonholonomic mobile robot in intelligent manufacturing environment | https://doi.org/10.1016/j.eswa.2021.116203 |
| 257 | 2024 | U-Net Semantic Segmentation-Based Calorific Value Estimation of Straw Multifuels for Combined Heat and Power Generation Processes | https://doi.org/10.3390/en17205143 |
| 294 | 2023 | Assembly performance evaluation method for prefabricated steel structures using deep learning and k-nearest neighbors | https://doi.org/10.12989/sss.2023.32.2.111 |
| 301 | 2023 | Automatic Inspection of Bridge Bolts Using Unmanned Aerial Vision and Adaptive Scale Unification-Based Deep Learning | https://doi.org/10.3390/rs15020328 |
| 307 | 2025 | CSSF-BL360: Color Segmentation-Based Bolt Localization and 0~360° Full-Scale Loosening Angle Detection in Single-Frame Images | https://doi.org/10.2174/0122127976329980240905064454 |
| 309 | 2023 | Comparison of deep learning and analytic image processing methods for autonomous inspection of railway bolts and clips | https://doi.org/10.1016/j.conbuildmat.2023.131472 |
| 310 | 2019 | Computer-vision based UAV inspection for steel bridge connections | https://doi.org/10.12783/shm2019/32473 |
| 320 | 2024 | Defect detection with adjustable template for screw hole checking | https://doi.org/10.1109/icce-taiwan62264.2024.10674316 |
| 330 | 2021 | Estimation of the gripping position and orientation of fasteners in camera images | https://doi.org/10.1109/ur52253.2021.9494679 |
| 341 | 2025 | Labelling and Counting of Bolts after Segmentation in a Multi Bolt Environment | https://doi.org/10.1109/icscc66177.2025.11233242 |
| 347 | 2025 | On the analysis of adapting deep learning methods to hyperspectral imaging. Use case for WEEE recycling and dataset | https://doi.org/10.1016/j.saa.2024.125665 |
| 351 | 2021 | Pandrol track fastener defect detection based on local convolutional neural networks | https://doi.org/10.1177/0959651820953679 |
| 359 | 2021 | Real-Time Defect Detection of Track Components: Considering Class Imbalance and Subtle Difference Between Classes | https://doi.org/10.1109/tim.2021.3117357 |
| 364 | 2026 | Research and Application of Multimodal Collaborative Intelligent Detection Technology and Maintenance Management & Control System for CRTS H Slab Ballastless Tracks | https://doi.org/10.3969/j.issn.1001-8360.2026.04.015 |
| 367 | 2019 | Research on segmentation of key parts for train safety based on deep learning | https://doi.org/10.1117/12.2548911 |
| 379 | 2025 | Versatile waste sorting in small batch and flexible manufacturing industries using deep learning techniques | https://doi.org/10.1038/s41598-025-87226-x |

## Selección de los 3 modelos base: U-Net, FCN y SegNet

Decisión tomada el 2026-11-07. Esta sección aplica, paso a paso, el procedimiento de extracción del PDF del Entregable 1 (sección "Modelos base", p. 13).

### Lo que exige el entregable

| Requisito del PDF (cita textual) | Cómo se cumple |
|---|---|
| Paso 1: "Revisar, en cada estudio incluido, qué modelos de Deep Learning se emplean como referencia o línea base." | Lectura de título y resumen de los 385 registros cribados; 206 incluidos. Confirmación en texto completo: pendiente (ver "Pendiente"). |
| Paso 3: "Construir una tabla de frecuencia que indique cuántos estudios utilizan cada modelo como referencia." | Tabla "Arquitecturas usadas o comparadas según el resumen" (arriba). |
| Paso 4: "Ordenar los candidatos por frecuencia de uso y por complejidad (número de parámetros y profundidad)." | Tabla de abajo. |
| Paso 5: "Seleccionar exactamente tres modelos: los más recurrentes y de menor complejidad arquitectónica, justificando la selección con las referencias que la respaldan." | U-Net, FCN y SegNet (justificación abajo). |
| Paso 6: "Verificar que los tres modelos no sean redundantes entre sí (deben diferir en el sesgo inductivo que incorporan)." | Ver "No redundancia". |
| Criterios: "Arquitectura simple, reproducible y ampliamente documentada"; "Uso frecuente como comparación en la literatura del problema elegido"; "Capacidad de entrenarse bajo el mismo pipeline y la misma partición". | Las tres se entrenan desde cero con el mismo pipeline, sin backbone preentrenado. |
| Modelos base típicos que el propio PDF indica para segmentación de imágenes: "red totalmente convolucional (FCN) sencilla, codificador–decodificador convolucional básico, U-Net reducida" (p. 14). | FCN = FCN sencilla; SegNet = codificador–decodificador convolucional básico; U-Net = U-Net reducida. |

### Paso 4: candidatos ordenados por frecuencia y complejidad

Frecuencia = estudios incluidos (206) que usan o comparan la arquitectura según el resumen. Parámetros = valores de los artículos originales (aproximados); **la tabla final debe recalcularse con la configuración que implemente el equipo**, porque el número de parámetros depende del ancho y del backbone.

| Candidato (semántica) | Frecuencia | Parámetros aprox. (versión original) | Componentes que añaden complejidad | ¿Pasa? |
|---|---|---|---|---|
| U-Net | 107 | ~31 M | ninguno extra: codificador–decodificador + saltos | Sí: la más frecuente; admite versión reducida |
| DeepLab (v3/v3+) | 30 | ~40–60 M según backbone | convoluciones dilatadas, ASPP, backbone profundo preentrenado | No: frecuente pero no es "de menor complejidad"; no figura en la lista de modelos base típicos del PDF |
| FCN | 13 | ~134 M (FCN-8s sobre VGG16); mucho menor en versión sencilla | ninguno extra: convoluciones + upsampling | Sí: tercera más frecuente; es la más simple conceptualmente |
| SegNet | 9 | ~29 M | ninguno extra: decodificador con índices de max-pooling | Sí: empata en frecuencia con PSPNet y U-Net++ y las supera en simplicidad o no redundancia |
| PSPNet | 9 | ~45–65 M según backbone | módulo de pooling piramidal, backbone preentrenado | No: misma frecuencia que SegNet, más compleja |
| U-Net++ | 9 | ~36 M | saltos anidados densos | No: variante de U-Net (redundante, paso 6) |

Se excluyen como candidatos YOLO-seg y Mask R-CNN: hacen segmentación de **instancias** (cada tornillo por separado), no la máscara binaria tornillo/fondo del problema (tipo 8, segmentación semántica).

### Justificación de cada modelo

- **U-Net**: la arquitectura más usada como referencia en el corpus (107 de 206 estudios), con mucha diferencia. Corresponde a la "U-Net reducida" del PDF.
- **FCN**: tercera semántica más frecuente (13) y la de menor complejidad arquitectónica. Corresponde a la "FCN sencilla" del PDF.
- **SegNet**: para el tercer puesto chocan los dos criterios del paso 5. DeepLab es más frecuente (30) pero más compleja (dilataciones, ASPP, backbone preentrenado). Entre las de menor complejidad, SegNet empata en frecuencia (9) con PSPNet y U-Net++; PSPNet es más compleja y U-Net++ es redundante con U-Net. SegNet corresponde al "codificador–decodificador convolucional básico" del PDF.

DeepLab y PSPNet quedan como candidatas para el Top 10 del Entregable 2, que pide modelos más avanzados.

### No redundancia (paso 6)

| Modelo | Sesgo inductivo (supuesto que la arquitectura impone) |
|---|---|
| FCN | La resolución se recupera con upsampling desde las capas profundas; poco detalle espacial fino. |
| SegNet | Decodificador simétrico que recupera la posición exacta usando los índices del max-pooling del codificador, sin copiar mapas de características. |
| U-Net | Conexiones de salto que copian mapas de características completos del codificador al decodificador, conservando detalle fino. |

Las tres difieren en **cómo recuperan la información espacial**, que es lo que más importa con tornillos pequeños (0.09 % de los píxeles, según el EDA).

## Pendiente

1. ✓ Cadenas: las de `revision_literatura_preliminar.md` (C1, C2, C3 por base). Búsqueda ejecutada el 2026-11-06 (hora de Colombia); Scopus marca la exportación como 07 November 2026.
2. ✓ Segunda pasada asistida por IA hecha. Los 24 dudosos están excluidos provisionalmente; **antes de subir el Entregable 1 (dom 2026-11-08)** el equipo los revisa a mano y, si alguno entra, se actualizan los conteos.
3. Elegibilidad por texto completo de los estudios que usarán para la tabla de síntesis (referencia, problema, dataset, modelos base, métricas, valores). Prioridad: los de segmentación semántica de tornillos y pernos listados abajo.
4. Evaluación de calidad de cada estudio incluido (protocolo, fuga de datos, repeticiones, pruebas estadísticas).

## Núcleo con "semantic segmentation" en título o resumen (lectura prioritaria)

| Año | Título | DOI | Arquitecturas en resumen |
|---|---|---|---|
| 2026 | DSRB-Net: A Super-Resolution and Boundary-Constrained Semantic Segmentation Method for Urban Rail Transit Bolt Anomaly Detection | https://doi.org/10.1109/jsen.2025.3642271 | — |
| 2026 | A Review of 3D Point Cloud Semantic Segmentation Methods for Complex Railway Scenes | https://doi.org/10.1061/jsued2.sueng-1702 | — |
| 2026 | Bolt-loosening visual detection method based on deep learning with super-resolution reconstruction and improved perspective transformation | https://doi.org/10.1088/2631-8695/ae76c2 | U-Net, YOLO(-seg) |
| 2026 | Research and Application of Multimodal Collaborative Intelligent Detection Technology and Maintenance Management & Control System for CRTS H Slab Ballastless Tracks | https://doi.org/10.3969/j.issn.1001-8360.2026.04.015 | YOLO(-seg) |
| 2026 | Profile detection of subway shield tunnels based on point cloud segmentation [基于点云分割的地铁盾构隧道轮廓检测技术研究] | https://doi.org/10.11817/j.issn.1672-7207.2026.05.008 | — |
| 2025 | Attention-guided triplet framework for synthetic data generation and semantic segmentation of railway fasteners under data scarcity and disparity | https://doi.org/10.1016/j.asoc.2025.113784 | — |
| 2025 | A Deep Learning Approach to Identify Rock Bolts in Complex 3D Point Clouds of Underground Mines Captured Using Mobile Laser Scanners | https://doi.org/10.3390/rs17152701 | — |
| 2025 | Labelling and Counting of Bolts after Segmentation in a Multi Bolt Environment | https://doi.org/10.1109/icscc66177.2025.11233242 | — |
| 2025 | Abnormality Detection of Screws Using Deep Learning and Image Processing | https://doi.org/10.1299/jsmelem.2025.11.os04-6 | DeepLab |
| 2025 | Defect Detection of Catenary Bolts Based on Dual-Cascade Architecture | https://doi.org/10.13238/j.issn.1004-2954.202310100005 | U-Net, YOLO(-seg) |
| 2024 | A Looseness Recognition Method for Rail Fastener Based on Semantic Segmentation and Fringe Projection Profilometry | https://doi.org/10.1109/jsen.2024.3389297 | — |
| 2024 | Shield Tunnel Dislocation Detection Method Based on Semantic Segmentation and Bolt Hole Positioning of MLS Point Cloud | https://doi.org/10.1109/tgrs.2024.3386308 | — |
| 2024 | Disassembly of Distribution Transformers Based on Multimodal Data Recognition and Collaborative Processing | https://doi.org/10.3390/a17120595 | YOLO(-seg) |
| 2024 | DEEP LEARNING-BASED RECOGNITION OF MANUFACTURING COMPONENTS USING AUGMENTED REALITY FOR WORKER TRAINING OF ASSEMBLY TASKS | https://doi.org/10.1115/msec2024-125279 | U-Net |
| 2024 | Evaluation of Methods for Detection and Semantic Segmentation of the Anterior Capsulotomy in Cataract Surgery Video | https://doi.org/10.2147/opth.s453073 | U-Net, ResNet (backbone) |
| 2024 | A Defect Detection Method Based on YOLOv7 for Automated Remanufacturing | https://doi.org/10.3390/app14135503 | YOLO(-seg) |
| 2024 | A modified U-Net CNN model for enhanced battery component segmentation in X-ray phone images | https://doi.org/10.1007/s00500-024-09831-8 | U-Net |
| 2024 | A Railway Inspection System Based on A Semantic Segmentation Network and Image Processing | https://doi.org/10.1109/icce-taiwan62264.2024.10674072 | U-Net |
| 2023 | RFS-Net: Railway Track Fastener Segmentation Network With Shape Guidance | https://doi.org/10.1109/tcsvt.2022.3212088 | — |
| 2023 | GL-Net: Semantic segmentation for point clouds of shield tunnel via global feature learning and local feature discriminative aggregation | https://doi.org/10.1016/j.isprsjprs.2023.04.011 | — |
| 2023 | Waysides Inspection Using Wayside Processing Imaging and Deep Learning | https://doi.org/10.1109/icpcsn58827.2023.00065 | U-Net, ResNet (backbone) |
| 2023 | Detection for Tiny Screw and Screw Hole by Semantic Segmentation Model | https://doi.org/10.1109/icmre56789.2023.10106576 | — |
| 2023 | IUCNet: A Novel Semantic Segmentation Network for Overlapping Industrial Parts Detection | https://doi.org/10.1109/cipae60493.2023.00078 | — |
| 2022 | Loosening Bolts Detection of Bogie Box in Metro Vehicles Based on Deep Learning | https://doi.org/10.1587/transinf.2022edl8041 | U-Net |
| 2021 | Research on Intelligent Detection of State of Catenary Puller Bolt Based on Deep Learning | https://doi.org/10.3969/j.issn.1001-8360.2021.11.007 | — |
| 2021 | A Deeplab-Based Segmentation Network for Screw Images | https://doi.org/10.1145/3453800.3453816 | DeepLab |
| 2021 | Estimation of the gripping position and orientation of fasteners in camera images | https://doi.org/10.1109/ur52253.2021.9494679 | — |
| 2020 | Real-Time Inspection System for Ballast Railway Fasteners Based on Point Cloud Deep Learning | https://doi.org/10.1109/access.2019.2961686 | — |
| 2018 | Deep learning based image recognition for crack and leakage defects of metro shield tunnel | https://doi.org/10.1016/j.tust.2018.04.002 | FCN |

Nota: incluye un falso positivo médico (cataratas) que el cribado manual debe excluir.
