# E2: búsqueda, cribado y selección del Top 10

De las búsquedas en Scopus, Web of Science e IEEE Xplore (2026-11-15) a los 10 modelos del E2.

## Cadena de trabajo

| Paso | Archivo | Qué contiene |
|---|---|---|
| 1. Cadenas | `cadenas_busqueda_E2.md` | Cadenas exactas: A1–A3 (problema) y B1–B4 (mecanismos transversales, PDF E2 §3.1). |
| 2. Registro | `registro_busquedas_E2.md` | Fecha, resultados por cadena y base, reglas de cribado, resolución de dudosos y cierre del Top 10. |
| 3. Cribado | `cribado_e2.py` + `decisiones_lectura_E2.csv` | Deduplica, aplica las reglas R0–R4 y toma de la tabla de decisiones lo que no sale de una regla automática. La tabla tiene la lectura de título y resumen (R5, asistida por IA), la revisión humana de los 12 dudosos y los mecanismos de la capa B. |
| | → `registros_unicos_E2.csv`, `prisma_E2.json` | Un registro único por fila con la regla que lo decidió; conteos del diagrama PRISMA. |
| | `prisma_e2.py` → `prisma_E2.png` | Diagrama PRISMA 2020 dibujado con esos conteos. |
| 4. Mecanismos | `tabla_mecanismos_E2.md` | Capa B: 56 registros en 6 mecanismos (M1–M6) y su transferibilidad a tornillos de ~10 px. |
| 5. Lista larga | `lista_larga_E2.csv`, `cadena_citas_originales.md` | 26 modelos: familia, citas (Scopus; Google Scholar para software sin paper), AP de máscara en COCO, implementación, filtros F1–F4. |
| 6. Puntaje | `puntaje_e2.py` → `puntaje_E2.csv` | Puntaje y selección con las reglas fijadas antes de calcular (hoja `reglas` de `lista_larga_puntaje_E2.xlsx`). |

## Conteos (salida de `cribado_e2.py`)

- Identificados: 1503 (Scopus 1031, WoS 372, IEEE 100). Duplicados: 440. Únicos: 1063.
- Excluidos por tipo de documento (portadas de actas, erratas): 38. Cribados: 1025.
- Capa A (problema): 811 cribados. R1 3D 99, R2 no menciona instancias 51, R3 sin métricas 158, R4 menos de 2 arquitecturas 423, R5 lectura 80 → **64 incluidos**.
- Capa B (mecanismos): 214 cribados → **56 incluidos**.

## Puntaje

PUNTAJE = 0.25 C + 0.15 R + 0.25 D + 0.20 I + 0.15 F

- **C, citación.** Citas por año: citas / (2026 − año del registro citado + 1). Se convierte en percentil dentro de la lista larga: la proporción de modelos con un valor estrictamente menor, dividida por n − 1.
- **R, recencia.** (año − mínimo) / (máximo − mínimo).
- **D, desempeño.** AP de máscara en COCO, escalado min-max.
- **I, implementación.** 1 si está en MMDetection 3.3 o ultralytics; 0.5 si solo hay repo oficial.
- **F, representatividad.** 1 si es el más citado por año de su familia; 0.5 si no.

Ejemplo, Mask2Former (puesto 1):

- 0.25 × 0.792 + 0.15 × 0.556 + 0.25 × 0.813 + 0.20 × 1 + 0.15 × 1 = **0.834**
- C = 0.792 sale de 19/24: 19 de los 25 modelos con citas tienen menos citas por año.

**Top 10:**

1. Mask2Former
2. YOLOv7-seg
3. YOLOv9-seg
4. YOLO26-seg
5. YOLO11-seg
6. Cascade Mask R-CNN
7. HTC
8. PointRend
9. Mask Scoring R-CNN
10. CondInst

- Mask DINO (puesto 6) no pasa F4.
- RTMDet-Ins sale por la regla de 4 familias.
- La justificación de cada modelo muy citado que queda fuera está al final de `registro_busquedas_E2.md`.
- Reservas: QueryInst y SCNet.
- Los modelos base del E1 (Mask R-CNN, SOLOv2, YOLOv8-seg) no compiten: van además del Top 10 (PDF E2 §3.2).

## Reproducir

```powershell
# exportaciones/: los 11 archivos tal como salen de las bases, con nombre "<cadena> - <base>.<ext>"
#   (p. ej. "A1 - scopus.csv", "A1 - wos.xls", "A1 - ieee.csv"). No se versionan (términos de uso).
& $PY literatura/e2/cribado_e2.py    # requiere pandas, xlrd y openpyxl
& $PY literatura/e2/prisma_e2.py     # requiere matplotlib
& $PY literatura/e2/puntaje_e2.py    # requiere pandas y numpy
```

## Limitaciones declaradas

- **R4 pierde estudios.** Excluye a los que no nombran arquitecturas en el resumen. En una muestra de 30 excluidos por R4 (semilla 42), 5 a 8 sí comparaban varias, o sea 17–27 %. La lista larga sale sobre todo de los surveys de A3, que R4 no filtra.
- **Exportación de IEEE.** Devolvió exactamente 100 registros para A1, lo que podría ser un tope de exportación.
- **Configuraciones COCO no idénticas.** El AP de COCO viene de la configuración de referencia de cada repositorio (R50 1x en MMDetection, "n" en ultralytics), así que no es exactamente comparable entre modelos.
