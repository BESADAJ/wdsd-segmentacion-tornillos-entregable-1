# Entrenamiento del E2

Código: `entrenar_e2.py` (orquestador) y `src/wdsd_e2/` (un trabajador por marco). Protocolo: `preregistro_E2.md`.

## Cómo funciona

- Cada modelo corre en el entorno de su marco. El orquestador solo usa la biblioteca estándar, así que se lanza con cualquier python. Cada corrida es un proceso aparte con el python que le toca:
  - `wdsd_dl`: YOLO11, YOLO26, YOLOv8, Mask R-CNN y el baseline clásico.
  - `wdsd_mmdet`: Mask2Former, Cascade, HTC, PointRend, MS R-CNN, CondInst y SOLOv2.
  - `wdsd_yolo9`: YOLOv7 y YOLOv9, con los repos oficiales.
- Una corrida terminada tiene `resumen.json`. Al relanzar un comando, se salta. Si se corta a la mitad (Ctrl+C o un apagón), la corrida empieza de cero la próxima vez.
- Si una corrida falla, se repite una vez con la misma semilla. Si vuelve a fallar, queda `fallo.json` y la cola sigue (preregistro §6). Cuentan como fallo:
  - una caída;
  - quedarse sin memoria;
  - una pérdida NaN o infinita;
  - que ultralytics baje el lote por su cuenta.
- La memoria de la GPU está topada al 95 %, igual que en la prueba de humo. Si no cabe, da error en vez de desbordarse a la RAM.

| Comando | Qué hace | Semilla y épocas | Salida |
|---|---|---|---|
| `configurar` | Comprueba los 3 entornos, los parches `nbs = 2` y los pesos de los repos YOLO. Guarda `maquina_e2.json`. | | `maquina_e2.json` (no se versiona) |
| `datos` | Crea la submuestra fija de 93 imágenes de train para la brecha de sobreajuste. | `random.Random(0)` | `datos_inst/coco/train93.json` y `datos_inst/yolo/train93.txt` |
| `piloto` | Mide tiempos y memoria. No predice test ni calcula F1. | 42, 1 época | `results/e2/piloto/<máquina>/` |
| `buscar` | Prueba T = 3 configuraciones por modelo y elige una por el F1 por grupo en val. | 42, ⌈E/3⌉ épocas | `results/e2/busqueda/<modelo>_c<valor>/` y `<modelo>_elegida.json` |
| `final` | Entrena con la configuración elegida y evalúa val, test y train93. Va por rondas: todos los modelos con una semilla, luego la siguiente. | las de `--semillas`, E épocas | `results/e2/corridas/<modelo>_s<semilla>/` |
| `resumen` | Junta todo en dos tablas. | | `results/e2/corridas.csv` y `results/e2/busqueda.csv` |

## Comandos (PowerShell, 3060)

```powershell
$T    = "D:\Documents\Personal\Uninorte\Maestría\Primer semestre\Deep Learning\Trabajo"
$RUN  = "$T\datos\wdsd"
$PY   = "C:\Users\ivanj\miniconda3\envs\wdsd_dl\python.exe"
$PYM  = "C:\Users\ivanj\miniconda3\envs\wdsd_mmdet\python.exe"
$PY9  = "C:\Users\ivanj\miniconda3\envs\wdsd_yolo9\python.exe"
$env:PYTHONUTF8 = "1"
Set-Location $RUN

# 1. Una vez por máquina
& $PY entrenar_e2.py configurar --nombre 3060 --py-dl $PY --py-mmdet $PYM --py-yolo9 $PY9 `
    --repo-yolov7 "$T\datos\yolov7u7\seg" --repo-yolov9 "$T\datos\yolov9"
& $PY entrenar_e2.py datos

# 2. Piloto de tiempo, en cada máquina (aprox. 1 h en la 3060)
& $PY entrenar_e2.py piloto --modelos todos

# 3. Después de la release del preregistro (E ya fijado)
& $PY entrenar_e2.py buscar --modelos <los de esta máquina> --epocas-final <E>
& $PY entrenar_e2.py final  --modelos todos --epocas <E> --semillas 42,43,44 --hasta "2026-11-18 22:00"
& $PY entrenar_e2.py resumen
```

`final` necesita el archivo `results/e2/busqueda/<modelo>_elegida.json` de cada modelo. Si la búsqueda de un modelo se hizo en la otra máquina, hay que copiar ese archivo antes.

## Pruebas en seco (CPU, datos sintéticos)

Se probó cada ruta de punta a punta en un entorno de CPU:

- Versiones: torch 2.1.2, mmcv 2.1.0 compilado sin CUDA, mmdet 3.3.0, ultralytics 8.4.207.
- Datos: 12 imágenes sintéticas de 1280×720.
- Opciones: `--permitir-cpu`, y `--sin-preentrenar` donde los pesos no se podían descargar.

Las rutas probadas:

- `configurar`, `datos`, `piloto`, `buscar` (con la elección de la configuración), `final` y `resumen`.
- Los 9 modelos de MMDetection, YOLO11, Mask R-CNN, el baseline clásico, la máscara vacía y YOLOv7/YOLOv9 con sus repos parchados.
- La repetición después de una falla.

Ese entorno valida el código, no los resultados: sin GPU y sin preentrenamiento las métricas no significan nada.

Dos cosas que salieron de la prueba:

- **QueryInst (reserva) falla solo en CPU.** Con pesos aleatorios, su cabeza de máscara recibe cajas degeneradas, y el pegado de máscaras de MMDetection en CPU falla. En GPU usa otro camino, y en la prueba de humo corrió bien.
- **La pérdida de YOLOv9-seg divide por cero.** En `utils/segment/loss_tal.py`, `target_scores_sum` no tiene `max(·, 1)`. Pasa cuando el asignador de etiquetas da puntaje 0 a todos los objetivos de una imagen: con imágenes sin tornillos y, como mostró el piloto real en la 3060, también al inicio del entrenamiento con tornillos de ≈ 10 px. El promedio de la pérdida de la época sale infinito. **Se parcha** con `max(target_scores.sum(), 1)`, que es lo que hace ultralytics en la misma pérdida (repos_oficiales/README.md, cambio 5). `m_oficial.py` verifica el parche antes de entrenar.
