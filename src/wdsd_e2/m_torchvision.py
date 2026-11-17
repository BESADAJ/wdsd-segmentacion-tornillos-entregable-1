"""Trabajador Mask R-CNN de torchvision (entorno wdsd_dl): el modelo base del E1, con su mismo código
(src/wdsd_inst/maskrcnn.py). Lo único que cambia es lo común del E2: tasa x factor, coseno hasta el 1 % y los splits
que se predicen (la búsqueda no toca test)."""
from __future__ import annotations

import time
from pathlib import Path

from . import registro as R
from .comun import DATOS, guardar_preds, pico_gb, ruta_json

LR_POR_DEFECTO = 0.0025 * R.LOTE_EFECTIVO  # receta de torchvision: 0.02 para lote 16, escalada linealmente


def entrenar(modelo: str, carpeta: Path, semilla: int, epocas: int, factor: float, splits: tuple, disp,
             preentrenado: bool = True, conservar_pesos: bool = True, log=print) -> dict:
    import torch
    from wdsd_inst import maskrcnn
    lote = R.MODELOS[modelo]["lote"]
    info = {"lr_inicial": LR_POR_DEFECTO * factor, "lr_por_defecto": LR_POR_DEFECTO, "optimizador": "SGD",
            "lote": lote, "acumulacion": 1}
    log(f"  {modelo}: {info}")
    t0 = time.time()
    r = maskrcnn.entrenar(DATOS, carpeta, semilla, epocas, disp, lote=lote, lr=LR_POR_DEFECTO * factor,
                          preentrenado=preentrenado, log=log, splits=(), lr_min_ratio=R.LR_FINAL)
    t_entrenar = time.time() - t0
    t1 = time.time()  # se predice aquí y no dentro de maskrcnn.entrenar para medir el tiempo y aplicar el filtro común
    m = maskrcnn.crear_modelo(preentrenado).to(disp)
    m.load_state_dict(torch.load(carpeta / "mejor.pt", map_location=disp))
    for split in splits:
        guardar_preds(carpeta, split, maskrcnn.predecir(m, ruta_json(split), disp))
    t_predecir = time.time() - t1
    if not conservar_pesos:
        (carpeta / "mejor.pt").unlink()
    return {**info, **pico_gb(), "t_entrenar_s": t_entrenar, "t_predecir_s": t_predecir,
            "mejor_epoca": r["mejor_epoca"], "parametros": r["parametros"], "historial": r["historial"]}
