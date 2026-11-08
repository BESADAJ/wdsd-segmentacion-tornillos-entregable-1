# Sección 3 del E1 (segmentación de instancias): texto para pegar en el notebook

Reemplaza la sección 3 semántica (celdas 44–52 de `01_eda_wdsd.ipynb`). Archivos que la acompañan:

- `prisma_instancias.py` (genera `prisma_instancias.png`)
- `tabla_sintesis_instancias.csv`
- `lectura_critica_instancias.md`

Etiquetas de verificación usadas al preparar el texto (no van en el notebook):
- **[O]** comprobado al preparar este texto, en la fuente citada o con el código.
- **[I]** inferido.

---

## 3. Revisión de literatura

### 3.1 Protocolo de búsqueda (PRISMA 2020)

- **Bases:** Web of Science (`TS=`), Scopus (`TITLE-ABS-KEY`) e IEEE Xplore ("All Metadata").
- **Fecha:** 2026-11-07. **Años:** 2018–2026.
- **Cadena C4 (Scopus):**

  ```
  TITLE-ABS-KEY("instance segmentation") AND TITLE-ABS-KEY(screw* OR bolt* OR fastener* OR disassembl* OR "e-waste" OR weee OR remanufactur* OR "industrial part*" OR "small object*") AND TITLE-ABS-KEY("deep learning" OR CNN OR "neural network*") AND TITLE-ABS-KEY(compar* OR baseline*) AND PUBYEAR > 2017
  ```

  En WoS se usa la misma cadena con `TS=` y `PY=(2018-2026)`.
- **Inclusión:**
  - segmentación de instancias con aprendizaje profundo;
  - dominio de tornillos, sujetadores, desmontaje o remanufactura;
  - compara al menos dos modelos de segmentación de instancias.
- **Exclusión:**
  - solo detección por cajas;
  - un solo modelo, sin comparación;
  - sin texto completo.
- **Otras fuentes:**
  - el artículo original del dataset (Kalitsios et al., 2022);
  - el único trabajo posterior que usa WDSD (Chao et al., 2025).
- **Proceso:**
  - Duplicados por DOI o, sin DOI, por título normalizado.
  - Primer cribado de título y resumen con un filtro de términos del dominio, registrado en `cribado_instancias_C4.csv`.
  - Revisión manual de los 17 registros del dominio y lectura de texto completo por el equipo.

| Etapa | n |
|---|---|
| Identificados (WoS 43, Scopus 62, IEEE 166) | 271 |
| Duplicados eliminados | 52 |
| Cribados por título y resumen | 219 |
| Excluidos por no tratar el dominio | 202 |
| Registros del dominio | 17 |
| Excluidos por resumen (fuera de dominio) | 2 |
| Buscados en texto completo | 15 |
| No recuperados (sin acceso al texto completo) | 4 |
| Evaluados en texto completo | 11 |
| Excluidos por usar un solo modelo | 4 |
| Incluidos de la búsqueda | 7 |
| Incluidos de otras fuentes | 2 |
| **Incluidos en la síntesis** | **9** |

Excluidos por resumen, por estar fuera de dominio: vértebras en rayos X (10.1007/s10278-026-02343-x) y huellas de edificios (10.12082/dqxxkx.2025.240628).

No recuperados:
- Bolt Sorting (10.1109/case59546.2024.10711806);
- grietas en remaches de tren (10.1038/s41598-024-61396-6);
- componentes de vía férrea (10.13382/j.jemi.b2407985);
- estado de sujetadores de vía con Mask R-CNN mejorada (10.3969/j.issn.1001-4632.2022.01.06).

### 3.2 Selección de los modelos base y tabla de síntesis

**Criterios:**
1. Frecuencia como comparador en la literatura.
2. Complejidad distinta entre sí.
3. No redundancia: cada modelo obtiene las instancias de una forma distinta.
4. Viables en una GPU de 6 GB.

**Frecuencia [O, `cribado_instancias_C4.csv` y `tabla_sintesis_instancias.csv`]:**

| Modelo | Resúmenes de la búsqueda (219) | Estudios de la síntesis (9) |
|---|---|---|
| YOLO (cualquier versión) | 64 | — |
| Mask R-CNN | 59 | 6 |
| YOLO-seg | 25 | 5 |
| SOLO / SOLOv2 | 10 | 2 |
| YOLACT | 7 | 3 |

**Terna elegida: Mask R-CNN, YOLOv8-seg y SOLOv2.** Cubre las tres familias:

| Modelo | Familia |
|---|---|
| Mask R-CNN | Dos etapas con cajas |
| YOLOv8-seg | Una etapa con cajas y prototipos |
| SOLOv2 | Una etapa sin cajas |

**Limitación declarada.** En texto completo YOLACT supera a SOLOv2 (3 estudios contra 2), pero en los resúmenes el orden se invierte (7 contra 10). Además, YOLACT comparte con YOLOv8-seg el mecanismo de prototipos, así que sería redundante.

**Equivalencia con la regresión logística.** Las tres cabezas de máscara terminan en una sigmoide por píxel sobre características aprendidas, es decir, en una regresión logística por píxel:
- Mask R-CNN y YOLOv8-seg la entrenan con entropía cruzada binaria, que es la verosimilitud de la regresión logística.
- SOLOv2 la entrena con pérdida Dice.

Por eso son la "línea base mínima razonable" del enunciado para este tipo de problema.

```python
tabla = pd.read_csv(f'{LIT}/tabla_sintesis_instancias.csv')
display(tabla[['referencia', 'problema', 'dataset', 'modelos_DL_de_referencia', 'metricas', 'valores_modelos_base']])
```

### 3.3 Modelos base

| Modelo | Artículo original | DOI o referencia |
|---|---|---|
| Mask R-CNN | He, Gkioxari, Dollár y Girshick (2017), ICCV, pp. 2980–2988 | 10.1109/ICCV.2017.322 [O, Crossref]. Versión TPAMI 42(2):386–397, 2020: 10.1109/TPAMI.2018.2844175 [O] |
| YOLOv8-seg | Jocher, Chaurasia y Qiu (2023), *Ultralytics YOLOv8*, versión 8.0.0, licencia AGPL-3.0 | No tiene artículo ni DOI; se cita como software: https://github.com/ultralytics/ultralytics [O, docs.ultralytics.com]. Mecanismo de prototipos: YOLACT, Bolya et al. (2019), ICCV, 10.1109/ICCV.2019.00925 [O] |
| SOLOv2 | Wang, Zhang, Kong, Li y Shen (2020), NeurIPS 33 | arXiv 2003.10152, 10.48550/arXiv.2003.10152 [O]. Las páginas del volumen no se verificaron. Base SOLO: Wang et al. (2020), ECCV, pp. 649–665, 10.1007/978-3-030-58523-5_38 [O] |

#### Mask R-CNN: dos etapas, con cajas

1. Una red de propuestas (RPN) sugiere regiones sobre un pirámide de características (ResNet-50-FPN).
2. RoIAlign recorta las características de cada región con interpolación bilineal, sin redondear coordenadas.
3. Tres cabezas en paralelo predicen clase, caja y una máscara de $m\times m$. En torchvision, $m=28$: RoIAlign de 14×14 seguido de una convolución transpuesta ×2 [O, código].

**Pérdida por región** [O, He et al. 2017, sec. 3]:

$$L = L_{cls} + L_{box} + L_{mask}, \qquad L_{mask} = -\frac{1}{m^2}\sum_{i,j}\big[y_{ij}\log\sigma(z_{ij}) + (1-y_{ij})\log(1-\sigma(z_{ij}))\big]$$

- $L_{mask}$ se calcula solo sobre el canal de la clase verdadera.
- torchvision suma además las dos pérdidas de la RPN (objetividad y regresión de la caja).

#### YOLOv8-seg: una etapa, con cajas y prototipos

En una sola pasada, la cabeza predice para cada posición:
- la caja;
- la clase;
- $k=32$ coeficientes de máscara.

En paralelo, una rama *Proto* genera 32 máscaras prototipo $P_k$ a 1/4 de la resolución de entrada. Toma P3 (paso 8) y lo sobremuestrea ×2 [O, `ultralytics/nn/modules/block.py`].

**Máscara de una detección:**

$$\hat m = \sigma\Big(\sum_{k=1}^{32} c_k P_k\Big)$$

recortada a su caja [O, `utils/loss.py`].

**Pérdida** [O, ultralytics 8.4.207, `utils/loss.py` y `cfg/default.yaml`]:

$$L = 7.5\,L_{CIoU} + 7.5\,L_{seg} + 0.5\,L_{cls} + 1.5\,L_{DFL}$$

- $L_{seg}$ es entropía cruzada binaria por píxel, recortada a la caja y normalizada por su área.
- $L_{cls}$ es entropía cruzada binaria.
- $L_{DFL}$ es la *distribution focal loss* de la caja.

#### SOLOv2: una etapa, sin cajas

1. La imagen se divide en una rejilla de $S\times S$ por nivel de la pirámide ($S$ = 40, 36, 24, 16, 12).
2. La celda que contiene el centro de un objeto es la responsable de ese objeto.
3. La rama de categoría predice un mapa $S\times S\times C$.
4. La rama de *kernel* predice, para cada celda, un filtro $G_{ij}$.
5. La máscara sale de una convolución dinámica, $M_{ij} = G_{ij} * F$ [O, Wang et al. 2020b, ec. 1], donde $F$ es una característica de máscara unificada a 1/4 de escala.
6. Las máscaras duplicadas se suprimen con Matrix NMS, que decae los scores en forma matricial con $f(\mathrm{iou}) = \exp(-\mathrm{iou}^2/\sigma)$ [O, apéndice].

**Pérdida** [O, SOLO, Wang et al. 2020a, ecs. 1–4; configuración oficial de MMDetection]:

$$L = L_{cate} + \lambda L_{mask},\quad \lambda = 3,\qquad L_{mask} = \frac{1}{N_{pos}}\sum_k \mathbb{1}\{p^*_{k} > 0\}\,\big(1 - D(p_k, q_k)\big),\qquad D(p,q) = \frac{2\sum p_{xy}q_{xy}}{\sum p_{xy}^2 + \sum q_{xy}^2}$$

$L_{cate}$ es una *focal loss* con $\gamma = 2$ y $\alpha = 0.25$.

#### Parámetros y costo

Conteo hecho con el código del proyecto, con 1 clase [O]:

| Modelo | Parámetros | Referencia publicada |
|---|---|---|
| Mask R-CNN R50-FPN (torchvision 0.29) | 43 922 395 | Con 91 clases da 44 401 393, igual a la documentación de torchvision (134.4 GFLOPs a 800 px) |
| YOLOv8n-seg (ultralytics 8.4.207) | 3 263 811 | 11.3 GFLOPs a 640 px, fusionado |
| SOLOv2 R50-FPN (MMDetection 3.3) | 46 228 801 | Comparte con Mask R-CNN el mismo ResNet-50-FPN (26.8 M parámetros) |

**Complejidad** [I, derivada de la arquitectura]. Para una imagen de $H\times W$:
- Cada convolución $k\times k$ de $C_{in}$ a $C_{out}$ cuesta $O(HW\,k^2C_{in}C_{out})$ en su resolución. Con la pirámide, el costo del extractor es $O(HW)$, con constantes distintas por modelo.
- **Mask R-CNN** suma un término por región: $O(HW + R\,m^2)$, con $R$ propuestas (512 por imagen en entrenamiento, hasta 1000 en inferencia).
- **YOLOv8-seg**: $O(HW + N\,k\,\tfrac{HW}{16})$ para ensamblar $N$ máscaras desde $k=32$ prototipos.
- **SOLOv2**: $O(HW + N\,E\,\tfrac{HW}{16})$ para la convolución dinámica de $N$ candidatos, más $O(N^2)$ en Matrix NMS.
- Las tres son lineales en el número de píxeles. Difieren en la constante y en cómo crecen con el número de instancias. El tiempo por época medido está en la sección 4.

### 3.4 Comparación crítica y vacío

Lectura completa en `lectura_critica_instancias.md`. Resumen:

- **No hay un orden estable entre los modelos.** En el único estudio con los tres modelos (AF-YOLO, objetos menores al 1 % de la imagen), el mAP de máscara queda en 83.2 (Mask R-CNN), 86.1 (SOLOv2) y 87.3 (YOLOv8-seg). En otro dominio, Mask R-CNN queda 12 puntos por debajo de los modelos de una etapa (SPGNet), y en conteo de sujetadores el orden cambia según la métrica.
- **Calidad metodológica de los 9 estudios:**
  - 0 usan más de una corrida;
  - 0 hacen prueba estadística;
  - 1 usa partición agrupada;
  - 3 aumentan datos antes de partir.
- **Los dos trabajos sobre WDSD:**
  - Kalitsios 2022: solo Mask R-CNN, partición oficial con fuga, AP 0.586.
  - Chao 2025: YOLOv8-seg, partición aleatoria tras aumentar datos.
- **Vacío:** ningún estudio compara Mask R-CNN, YOLOv8-seg y SOLOv2 en segmentación de instancias de tornillos con partición agrupada, varias semillas e inferencia estadística. Este trabajo cubre ese vacío.
