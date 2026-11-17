"""Utilidades comunes a los trabajadores del E2 (corren dentro del entorno de cada marco)."""
from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "datos_inst"
FRACCION_VRAM = 0.95   # tope del asignador de PyTorch: sin él, Windows desborda a RAM sin dar error (prueba de humo v2)
SCORE_MIN = 0.05       # mismo filtro de inferencia que el E1
MAX_DET = 100


def ruta_json(split: str) -> Path:
    return DATOS / "coco" / f"{split}.json"


def carpeta_imagenes(split: str) -> Path:
    """train93 es una submuestra de train: sus imágenes están en images/train."""
    return DATOS / "yolo" / "images" / ("train" if split == "train93" else split)


def imagenes(split: str) -> list[dict]:
    return json.loads(ruta_json(split).read_text(encoding="utf8"))["images"]


def n_train(con_anotaciones: bool = False) -> int:
    d = json.loads(ruta_json("train").read_text(encoding="utf8"))
    if not con_anotaciones:
        return len(d["images"])
    return len({a["image_id"] for a in d["annotations"]})


def fijar_semilla(s: int):
    import numpy as np
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def preparar_gpu(permitir_cpu: bool):
    import torch
    if not torch.cuda.is_available():
        if not permitir_cpu:
            raise SystemExit(f"torch {torch.__version__} no ve la GPU. Revisa el entorno (o --permitir-cpu solo para pruebas).")
        return torch.device("cpu")
    torch.cuda.set_per_process_memory_fraction(FRACCION_VRAM, 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.device("cuda:0")


def pico_gb() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"gpu_pico_asignada_gb": None, "gpu_pico_reservada_gb": None}
    return {"gpu_pico_asignada_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "gpu_pico_reservada_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)}


def entorno() -> dict:
    d = {"python": platform.python_version(), "so": platform.platform()}
    try:
        import torch
        d.update(torch=torch.__version__,
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except ImportError:
        pass
    for mod in ("torchvision", "ultralytics", "mmdet", "mmcv", "mmengine", "numpy"):
        try:
            d[mod] = __import__(mod).__version__
        except Exception:
            pass
    return d


def filtrar(preds: list[dict]) -> list[dict]:
    """Mismo posproceso para todos: score >= 0.05 y como mucho 100 detecciones por imagen (las de mayor score)."""
    por_img: dict = {}
    for p in preds:
        if p["score"] >= SCORE_MIN:
            por_img.setdefault(p["image_id"], []).append(p)
    salida = []
    for v in por_img.values():
        salida += sorted(v, key=lambda p: -p["score"])[:MAX_DET]
    return salida


def guardar_preds(carpeta: Path, split: str, preds: list[dict]):
    (carpeta / f"preds_{split}.json").write_text(json.dumps(filtrar(preds)))


def escribir_json(ruta: Path, obj):
    ruta.write_text(json.dumps(obj, indent=1, default=str), encoding="utf8")
