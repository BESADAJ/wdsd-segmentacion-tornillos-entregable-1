# Entrenamiento de los modelos base (FCN, SegNet, U-Net)

## Qué hay

| Archivo | Qué hace |
|---|---|
| `entrenar_base.py` | Punto de entrada. Se corre desde la raíz del repositorio. |
| `src/wdsd_e1/modelos.py` | Los tres modelos. Comparten el mismo codificador; solo cambia cómo recuperan la resolución (ese es su sesgo inductivo). |
| `src/wdsd_e1/entrenar.py` | Carga de datos, entrenamiento, evaluación, piloto y cola nocturna. |

Usa la partición agrupada (`results/eda/particion.csv`, columna `split`) y la normalización de train (`results/eda/normalizacion_train.json`) que ya produjo el EDA.

## 1. Entorno aislado (una sola vez, ~15 min)

En **Anaconda Prompt** (no en la terminal de VS Code, porque conda no está en su PATH):

```
conda create -n wdsd_dl python=3.11 -y
conda activate wdsd_dl
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pillow
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

La última línea debe imprimir `True` y el nombre de la tarjeta (RTX 3060 o 3050). Si imprime `False`, **no lances la cola**: el entrenamiento correría en CPU y no termina.

Para ver la ruta del Python del entorno (la necesitas en VS Code): `where python` con el entorno activado. Suele ser `C:\Users\<usuario>\anaconda3\envs\wdsd_dl\python.exe` (o `miniconda3`).

## 2. Dónde van los datos

Igual que el EDA: las carpetas descargadas `WDSD-*/WDSD` van en la carpeta **padre** del repositorio. Si están en otro lado, añade `--data-root "D:\ruta\a\la\carpeta"` a los comandos.

## 3. Piloto (~10–20 min)

En la terminal de VS Code, desde la raíz del repositorio:

```
& "C:\...\envs\wdsd_dl\python.exe" entrenar_base.py piloto --horas 6.5
```

Entrena 1 época de cada modelo con la semilla 42 y muestra:
- parámetros y número de capas convolucionales de cada modelo (sirve para la tabla de complejidad del paso 4 del PDF);
- segundos por época y memoria de GPU;
- cuántas épocas caben para 3 modelos × 5 semillas en las horas indicadas, y el comando sugerido.

## 3b. Curva de aprendizaje (opcional, para decidir la partición)

```
& "C:\...\envs\wdsd_dl\python.exe" entrenar_base.py curva --hasta 07:20 --max-epocas N --data-root .
```

Entrena U-Net (semilla 42) con el 100 %, 50 % y 75 % de los **grupos** de train (subconjuntos anidados) y reporta el mejor Dice de **validación** de cada uno en `results/modelos_base/curva/curva.csv`. Reparte el tiempo restante por igual entre las fracciones pendientes; si una se corta por tiempo, la tabla lo marca (`cortada_por_tiempo`). Si se interrumpe, al relanzar salta lo terminado.

Lectura: si el Dice sigue subiendo claramente de 75 % a 100 %, quitar grupos de train para ampliar test costaría desempeño; si se aplana, cuesta poco.

## 4. Cola nocturna

Antes: computador **conectado a la corriente**, suspensión desactivada (`powercfg /change standby-timeout-ac 0` en una terminal como administrador, o en Configuración > Energía), y cerrar programas pesados.

```
& "C:\...\envs\wdsd_dl\python.exe" entrenar_base.py cola --hasta 07:15 --max-epocas N
```

`N` = el que sugiere el piloto (máximo 40). La cola:
- corre por rondas: semilla 42 de FCN, SegNet y U-Net; luego 43; etc. Si el tiempo no alcanza, quedan los 3 modelos con el mismo número de semillas (±1);
- antes de cada corrida estima si termina antes de `--hasta`; si no, se detiene sin empezarla;
- si se interrumpe, al relanzar el mismo comando salta lo que ya terminó.

## 5. Qué produce (`results/modelos_base/`)

| Archivo | Contenido |
|---|---|
| `corridas.csv` | Una fila por corrida: parámetros, capas, épocas, Dice val y test, tiempos, memoria pico. |
| `<modelo>_s<semilla>/test_por_imagen.csv` | Dice, IoU, precisión y recall de píxel **por imagen de prueba**, con su grupo. Es la base de las pruebas estadísticas (pareadas por imagen o por grupo). |
| `<modelo>_s<semilla>/historial.csv` | Pérdida y Dice de validación por época (curvas de aprendizaje). |
| `<modelo>_s<semilla>/resumen.json` | Todo lo anterior + configuración y versiones (reproducibilidad). |
| `<modelo>_s<semilla>/mejor.pt` | Pesos del mejor epoch. |

## Protocolo (igual para los tres)

- Recortes de 512×512 a resolución nativa (el EDA mostró que reducir a 256 limita el Dice a 0.84). La mitad de los recortes se centra cerca de un tornillo, porque los tornillos son ~0.09 % de los píxeles.
- Volteos horizontal y vertical como aumento de datos.
- Pérdida 0.5 × entropía cruzada binaria + 0.5 × Dice suave. Adam, lr 1e-3 con coseno, lote 8, precisión mixta.
- Validación y prueba sobre la imagen completa 720×1280; métrica: Dice de tornillo por imagen (imagen sin tornillos y sin predicción = 1).
- Mejor epoch por Dice medio de validación; parada temprana con paciencia 8. La prueba se evalúa una sola vez por corrida.
- Semillas 42–46. cuDNN en modo determinista (`deterministic=True`, `benchmark=False`). En GPU, el gradiente del sobremuestreo bilineal (FCN) no es exactamente determinista, así que repetir una misma semilla puede dar diferencias mínimas en los decimales.
- Ancho base 16 ("U-Net reducida"). Mismo codificador en los tres modelos: FCN no tiene decodificador aprendido, SegNet desagrupa con índices del max-pooling, U-Net concatena los mapas del codificador.
