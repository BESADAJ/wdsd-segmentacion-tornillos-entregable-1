# Reproducibilidad

Para repetir el análisis y los entrenamientos, en este anexo se indica de dónde se obtienen los datos, qué entornos se usaron, qué comandos se ejecutan y qué deja cada corrida. Para leer el libro no hace falta nada de esto, porque sus páginas usan las salidas ya guardadas en los notebooks.

## Datos

El dataset no está en el repositorio: pesa 2,7 GB y las fuentes públicas no especifican una licencia de uso o redistribución.

1. Descargar la carpeta `WDSD` desde la [página oficial](https://vcl.iti.gr/dataset/weee-disassembly-screw-dataset/). Llega en dos archivos zip y hacen falta los dos: cada uno trae cerca de la mitad de las imágenes y de las máscaras, y el COCO de entrenamiento viene en el segundo.
2. Extraer ambos junto al repositorio:

```text
carpeta_de_trabajo/
├── WDSD-<fecha>-1-001/WDSD/
├── WDSD-<fecha>-1-002/WDSD/
└── wdsd-segmentacion-tornillos-entregable-1/
```

Deben quedar 945 pares de imagen y máscara y 4.398 instancias COCO. La huella SHA-256 esperada del dataset está en `results/eda/declaracion_a_priori.json`.

## Partición y métrica

- La partición agrupada congelada está en `results/eda/particion.csv`: 744, 93 y 108 imágenes (3.530, 479 y 389 tornillos), con 146, 26 y 14 grupos de escena. No se regenera ni se consulta el conjunto de prueba durante el ajuste.
- La verdad de referencia son los polígonos COCO, rasterizados por centros de píxel para reproducir el campo `area`.
- La métrica primaria es el F1 de instancia por imagen con IoU ≥ 0,5. Su declaración, junto con δ = 0,20, está en `results/eda/declaracion_a_priori_instancias.json`.
- La normalización usa la media y la desviación de entrenamiento guardadas en `results/eda/normalizacion_train.json`.

## Entornos

| Uso | Python | Paquetes principales | Archivo de dependencias |
|---|---|---|---|
| EDA (capítulos 1 y 2) | 3.9 | numpy, pandas, scipy, scikit-image, pycocotools | `requirements.txt` |
| Mask R-CNN y YOLOv8-seg | 3.11 | torch 2.11 (CUDA 12.8), torchvision 0.26, ultralytics 8.4.207 | `requirements_wdsd_dl_congelado.txt` |
| SOLOv2 | 3.11 | torch 2.1.2 (CUDA 12.1), mmcv 2.1.0, MMDetection 3.3.0 | `requirements_solov2.txt` |

SOLOv2 va en un entorno aparte porque mmcv 2.1 solo tiene versiones precompiladas para Windows con torch 2.1.

## Entrenamiento

Los scripts se niegan a entrenar si no detectan la GPU. Los pasos, en orden:

1. **Preparar los datos.** `python entrenar_instancias.py preparar --data-root <carpeta de WDSD>` reagrupa el COCO con la partición congelada y genera el formato YOLO. Debe imprimir 744/93/108 imágenes y 3530/479/389 instancias.
2. **Piloto de tiempos.** `python entrenar_instancias.py piloto` y `python solov2_instancias.py piloto` entrenan una época de cada modelo con la semilla 42 e informan los segundos por época, la memoria pico y cuántas épocas caben en el tiempo disponible. Con eso se fijaron 12, 40 y 12 épocas.
3. **Cola de la terna.** `cola_terna.ps1` entrena por rondas de semilla con los tres modelos (semilla 42 de los tres, luego 43, y así), de modo que si el tiempo se acaba todos quedan con el mismo número de semillas:

```powershell
powershell -ExecutionPolicy Bypass -File .\cola_terna.ps1 -Hasta 18:00 -EpMaskRcnn 12 -EpYolo 40 -EpSolo 12 -Semillas 42,43,44
```

Los lotes por defecto son 2, 4 y 2, iguales en todas las máquinas. El reparto se hizo por semilla: 42, 43 y 44 en una RTX 3060 Laptop de 6 GB; 45 y 46 en una RTX 3050 Laptop de 6 GB. Si la cola se corta, al relanzarla salta las corridas terminadas.

4. **Resumen y estadística.** `python entrenar_instancias.py resumen` escribe `results/instancias/resumen.csv`, y `estadistica_instancias.py` hace la comparación por grupos del capítulo 5 (Friedman, Wilcoxon con Holm, bootstrap y efecto mínimo detectable).

## Qué deja cada corrida

Cada corrida escribe en `results/instancias/corridas/<modelo>_s<semilla>/`:

| Archivo | Contenido |
|---|---|
| `resumen.json` | Entorno, tiempo, memoria GPU, número de parámetros y métricas |
| `metricas.json` | AP, AP50, AP75 y AP *small* de máscara en validación y prueba; F1, PQ, SQ y RQ por imagen; confianza mínima elegida en validación |
| `test_por_imagen.csv` | Una fila por imagen de prueba con su grupo, TP, FP, FN, F1 y PQ |
| `preds_val.json`, `preds_test.json` | Predicciones en formato COCO (RLE), para recalcular cualquier métrica sin reentrenar |

Los pesos entrenados no se versionan.

## Construir el libro

```bash
pip install "jupyter-book==1.0.4.post1"
jupyter-book build .
```

El libro queda en `_build/html/index.html`. No necesita el dataset ni una GPU, porque los notebooks ya están ejecutados. Cada cambio en la rama `main` lo vuelve a construir y publicar automáticamente.
