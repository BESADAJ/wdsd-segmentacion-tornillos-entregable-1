"""YOLOv8-seg (ultralytics, preentrenado en COCO) para tornillos.

- imgsz=1280: a 640 el diámetro mediano del tornillo baja de 11.7 a 5.9 px y el 79 % queda bajo 8 px (EDA).
- La mejor época la elige ultralytics con su "fitness" (combinación de mAP de caja y de máscara en val).
- Las predicciones de val y test se exportan al formato común (RLE) y se evalúan con metricas.py,
  igual que Mask R-CNN.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .metricas import a_rle


def entrenar(datos: Path, carpeta: Path, semilla: int, epocas: int, disp, variante: str = "yolov8n-seg",
             lote: int = 4, imgsz: int = 1280, preentrenado: bool = True, fraccion: float = 1.0, log=print) -> dict:
    from ultralytics import YOLO

    carpeta.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(f"{variante}.pt" if preentrenado else f"{variante}.yaml")
    if disp.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    modelo.train(data=str(datos / "yolo/data.yaml"), imgsz=imgsz, epochs=epocas, batch=lote, seed=semilla,
                 deterministic=True, workers=2, device=0 if disp.type == "cuda" else "cpu",
                 project=str(carpeta.resolve()), name="ultra", exist_ok=True, plots=False, verbose=False,
                 fraction=fraccion, amp=disp.type == "cuda", pretrained=preentrenado, patience=max(10, epocas))
    t_total = time.time() - t0
    pico = torch.cuda.max_memory_allocated() / 2**30 if disp.type == "cuda" else float("nan")
    pesos = carpeta / "ultra" / "weights" / "best.pt"
    mejor = YOLO(str(pesos))
    for split in ("val", "test"):
        (carpeta / f"preds_{split}.json").write_text(json.dumps(predecir(mejor, datos, split, imgsz, disp)))
    hist = carpeta / "ultra" / "results.csv"
    return {"s_total_entrenamiento": t_total, "s_por_epoca": t_total / epocas, "gpu_pico_gb": pico,
            "parametros": sum(p.numel() for p in mejor.model.parameters()),
            "historial_csv": str(hist) if hist.exists() else None}


def predecir(modelo, datos: Path, split: str, imgsz: int, disp) -> list[dict]:
    gt = json.loads((datos / f"coco/{split}.json").read_text(encoding="utf8"))
    salida = []
    for im in gt["images"]:
        ruta = datos / "yolo/images" / split / f"{Path(im['file_name']).stem}.png"
        r = modelo.predict(str(ruta), imgsz=imgsz, conf=0.05, iou=0.6, max_det=100, retina_masks=True,
                           device=0 if disp.type == "cuda" else "cpu", verbose=False)[0]
        if r.masks is None:
            continue
        ms = r.masks.data.cpu().numpy() > 0.5  # (n, H, W) a resolución original con retina_masks=True
        sc = r.boxes.conf.cpu().numpy()
        for s, m in zip(sc, ms):
            if m.shape != (im["height"], im["width"]):
                raise ValueError(f"Máscara de forma {m.shape}, esperaba {(im['height'], im['width'])}")
            if m.any():
                salida.append({"image_id": im["id"], "category_id": 1, "segmentation": a_rle(m), "score": float(s)})
    return salida
