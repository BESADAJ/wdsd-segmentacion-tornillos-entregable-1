# Cadenas de búsqueda del E2 (fijadas y ejecutadas el 2026-11-15)

Requisitos del PDF del E2 §3.1 [observado]: al menos dos bases entre WoS, Scopus e IEEE Xplore (arXiv solo complementaria); registrar la fecha y el número de resultados de cada cadena; periodo de 5 a 8 años; además, **al menos tres cadenas de "búsqueda transversal de mecanismos"** que combinen (i) familia del mecanismo, (ii) modalidad distinta al problema y (iii) "transfer" o "adaptation".

Estado: ejecutadas el 2026-11-15; resultados por cadena y base en `registro_busquedas_E2.md`. Regla fijada antes de ejecutar: si una devuelve más de ~1000 o menos de ~20, se ajusta **antes del cribado** y el ajuste se registra con su fecha.

## A. Problema (amplía la C4 del E1)

La C4 exigía dominio de tornillos/desmontaje Y "compar* OR baseline*". Aquí se quita ese filtro y se añaden objetos pequeños e inspección industrial, para encontrar modelos y no solo aplicaciones.

**A1. Objetos pequeños y sujetadores**
```
Scopus: TITLE-ABS-KEY("instance segmentation") AND TITLE-ABS-KEY(screw* OR bolt* OR fastener* OR rivet* OR "small object*" OR "tiny object*") AND TITLE-ABS-KEY("deep learning" OR "convolutional neural network*" OR transformer*) AND PUBYEAR > 2017 AND PUBYEAR < 2027
WoS:    TS=("instance segmentation") AND TS=(screw* OR bolt* OR fastener* OR rivet* OR "small object*" OR "tiny object*") AND TS=("deep learning" OR "convolutional neural network*" OR transformer*) AND PY=(2018-2026)
IEEE:   ("All Metadata":"instance segmentation") AND ("All Metadata":screw* OR "All Metadata":bolt* OR "All Metadata":fastener* OR "All Metadata":"small object*" OR "All Metadata":"tiny object*")   [filtro de años 2018-2026 en la interfaz]
```

**A2. Inspección industrial y desmontaje**
```
Scopus: TITLE-ABS-KEY("instance segmentation") AND TITLE-ABS-KEY(industrial OR manufactur* OR "electronic component*" OR PCB OR "e-waste" OR WEEE OR disassembl* OR remanufactur*) AND PUBYEAR > 2017 AND PUBYEAR < 2027
WoS:    TS=("instance segmentation") AND TS=(industrial OR manufactur* OR "electronic component*" OR PCB OR "e-waste" OR WEEE OR disassembl* OR remanufactur*) AND PY=(2018-2026)
```

**A3. Arquitecturas generales (surveys y benchmarks), para la lista larga y las familias**
```
Scopus: TITLE("instance segmentation") AND TITLE(survey OR review OR benchmark*) AND PUBYEAR > 2017 AND PUBYEAR < 2027
WoS:    TI=("instance segmentation") AND TI=(survey OR review OR benchmark*) AND PY=(2018-2026)
arXiv (complementaria): ti:"instance segmentation" AND (ti:survey OR ti:review)
```
Los trabajos fundacionales anteriores a 2018 (por ejemplo, el artículo original de Mask R-CNN, 2017) se incluyen por esta vía y se justifican como fundacionales, como permite el PDF.

## B. Búsqueda transversal de mecanismos (mínimo 3)

Cada una combina familia del mecanismo + modalidad distinta + transferencia/adaptación. Están pensadas para alimentar el E3: el modelo base más débil (SOLOv2) falla por la asignación de objetos de ~11 px (inferido), y el grupo 9 mezcla tornillos del dispositivo con tornillos del entorno.

**B1. Asignación de etiquetas para objetos diminutos (aéreas, microscopía)**
```
Scopus: TITLE-ABS-KEY("label assignment" OR "sample assignment" OR "center sampling" OR "assignment strategy") AND TITLE-ABS-KEY("tiny object*" OR "small object*") AND TITLE-ABS-KEY(aerial OR "remote sensing" OR microscop* OR cell*) AND TITLE-ABS-KEY(transfer* OR adapt*) AND PUBYEAR > 2017
```

**B2. Convolución dinámica / kernels condicionales (médica, teledetección)**
```
Scopus: TITLE-ABS-KEY("dynamic convolution" OR "dynamic kernel*" OR "conditional convolution*") AND TITLE-ABS-KEY(segmentation) AND TITLE-ABS-KEY(medical OR "remote sensing" OR satellite) AND TITLE-ABS-KEY(transfer* OR adapt*) AND PUBYEAR > 2017
```

**B3. Atención basada en consultas / transformers de máscara (médica, video, teledetección)**
```
Scopus: TITLE-ABS-KEY("query-based" OR "mask transformer" OR "masked attention" OR "object quer*") AND TITLE-ABS-KEY(segmentation) AND TITLE-ABS-KEY(medical OR video OR "remote sensing") AND TITLE-ABS-KEY(transfer* OR adapt*) AND PUBYEAR > 2017
```

**B4 (opcional). Contexto objeto–entorno (distinguir tornillo del dispositivo de tornillo del banco)**
```
Scopus: TITLE-ABS-KEY("context modeling" OR "contextual reasoning" OR "scene context") AND TITLE-ABS-KEY("instance segmentation" OR "object detection") AND TITLE-ABS-KEY("remote sensing" OR medical OR "autonomous driving") AND TITLE-ABS-KEY(transfer* OR adapt*) AND PUBYEAR > 2017
```

Las cadenas de WoS se obtienen cambiando `TITLE-ABS-KEY(` por `TS=(` y `PUBYEAR > 2017` por `PY=(2018-2026)`.

## Registro por cadena
Fecha de ejecución, número de resultados por cadena y base, y ajustes: `registro_busquedas_E2.md`.
