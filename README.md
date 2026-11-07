# WDSD: segmentación de tornillos, entregable 1 de Deep Learning

EDA del WEEE Disassembly Screw Dataset (WDSD). El EDA está terminado; la revisión de literatura, los tres modelos base y el análisis estadístico están pendientes.

**Empieza por [`notebooks/01_eda_wdsd.ipynb`](notebooks/01_eda_wdsd.ipynb).** Ya está ejecutado, así que se puede leer sin descargar los datos. Su sección 13 resume el veredicto y las decisiones.

## Estado del entregable

| Sección del enunciado | Peso | Estado |
|---|---|---|
| 1. Problema y dataset | parte del EDA | Hecho, con pendientes abajo |
| 2. EDA | 30 % | Hecho |
| 3. Literatura y tres modelos base | 30 % | Pendiente |
| 4. Implementación de los modelos | 20 % | Pendiente |
| 5. Benchmark, baseline y estadística | 20 % | Pendiente |

## Lo que debes saber antes de modelar

1. **No uses la partición oficial como principal.** Tiene fuga: el 48,9 % de test tiene una casi-gemela en train. Usa `results/eda/particion.csv` (columna `split`), con 744 / 93 / 108 imágenes. Evalúa la oficial solo como análisis secundario, para comparar con la literatura.
2. **No regeneres la partición ni consultes test** mientras ajustas. El CSV y sus hashes en `declaracion_a_priori.json` son la referencia congelada.
3. **No midas con exactitud de píxel.** Predecir todo fondo da 99,9 % y Dice 0. La métrica primaria es el Dice de tornillo promediado por imagen, y $\delta = 0{,}02$ es provisional hasta el piloto.
4. **No reduzcas la imagen completa.** Los tornillos miden 11,5 px de diámetro mediano. Entrena con parches de 512×512 a resolución nativa y evalúa a 1280×720 con ventana deslizante, con solape de al menos 40 px.
5. **Normaliza con** `results/eda/normalizacion_train.json`.
6. **Elimina las componentes de menos de 5 px** de las máscaras: son 4 artefactos de 1 px en todo el dataset.
7. **Ten cuidado con la inferencia.** Test solo tiene 14 grupos independientes, así que el bootstrap debería remuestrear grupos además de imágenes.

## Pendientes

- [ ] Confirmar la licencia con los autores: la página oficial no la declara.
- [ ] Conseguir el DOI del artículo de referencia (Kalitsios et al., ICRCV 2022).
- [ ] Auditar una muestra para estimar tornillos sin anotar.
- [ ] Comprobar si la literatura sobre WDSD usa la partición oficial: podría ser el *gap* a formular.
- [ ] Preparar un entorno con GPU. En el entorno usado para el EDA, TensorFlow 2.10 no detecta la GPU y PyTorch no está instalado.

## Datos

El dataset no está en este repositorio porque pesa 2,7 GB y no tiene licencia declarada.

1. Descarga la carpeta `WDSD` desde la [página oficial](https://vcl.iti.gr/dataset/weee-disassembly-screw-dataset/). Google Drive la entrega en dos archivos zip, y **hacen falta los dos**: cada uno trae cerca de la mitad de las imágenes y de las máscaras, sin que coincidan entre sí.
2. Extrae ambos junto a este repositorio:

```text
carpeta_de_trabajo/
├── WDSD-<fecha>-1-001/WDSD/
├── WDSD-<fecha>-1-002/WDSD/
└── wdsd-segmentacion-tornillos-entregable-1/
```

Debes tener 945 pares. El SHA-256 esperado del dataset, según `E.huella_dataset`, está en `results/eda/declaracion_a_priori.json`.

## Ejecución

Requiere Python 3.9:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_wdsd.ipynb
```

La primera ejecución completa tarda unos 20 minutos, porque calcula hashes y medidas por imagen. Esos cálculos quedan en caché en `results/eda/` y no se versionan.

## Uso de la partición

```python
import pandas as pd

particion = pd.read_csv("results/eda/particion.csv")
train = particion[particion.split == "train"]
```

La columna `grupo` identifica la unidad de independencia y `familia` la familia de adquisición. El módulo `src/wdsd_e1/eda.py` reúne el inventario y las medidas usadas en el EDA.
