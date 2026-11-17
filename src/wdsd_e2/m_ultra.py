"""Trabajador ultralytics 8.4.207 (entorno wdsd_dl): YOLO11n-seg, YOLO26n-seg y YOLOv8n-seg (modelo base del E1).

- Optimizador: el que elige optimizer=auto con 1 clase y < 10 000 pasos (AdamW, lr0 = 0.002, momentum 0.9,
  warmup_bias_lr 0), fijado explícitamente para que no cambie con el número de épocas.
- nbs = 2 (lote efectivo 2, sin acumulación con lote 2). Única aumentación: volteo horizontal 0.5.
- Coseno por época hasta lrf = 0.01; calentamiento común del E2 expresado en épocas.
- Mejor época: AP de máscara 0.5:0.95 en val (columna metrics/mAP50-95(M) de results.csv), no el "fitness" de
  ultralytics (caja + máscara). save_period=1 guarda epoch{k}.pt con k desde 0; results.csv numera desde 1.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

from . import registro as R
from .comun import DATOS, carpeta_imagenes, guardar_preds, imagenes, n_train, pico_gb

AUMENTACION = dict(mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0,
                   translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0,
                   close_mosaic=0, multi_scale=0.0)
LR_POR_DEFECTO = 0.002


def leer_results(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf8") as f:
        return [{k.strip(): float(v) for k, v in fila.items() if k is not None and v not in (None, "")}
                for fila in csv.DictReader(f)]


def predecir(yolo, split: str, disp) -> list[dict]:
    from wdsd_inst.metricas import a_rle
    salida = []
    for im in imagenes(split):
        ruta = carpeta_imagenes(split) / f"{Path(im['file_name']).stem}.png"
        r = yolo.predict(str(ruta), imgsz=1280, conf=0.05, iou=0.6, max_det=100, retina_masks=True,
                         device=0 if disp.type == "cuda" else "cpu", verbose=False)[0]
        if r.masks is None:
            continue
        ms = r.masks.data.cpu().numpy() > 0.5
        sc = r.boxes.conf.cpu().numpy()
        for s, m in zip(sc, ms):
            if m.shape != (im["height"], im["width"]):
                raise ValueError(f"Máscara de forma {m.shape}, esperaba {(im['height'], im['width'])}")
            if m.any():
                salida.append({"image_id": im["id"], "category_id": 1, "segmentation": a_rle(m), "score": float(s)})
    return salida


def entrenar(modelo: str, carpeta: Path, semilla: int, epocas: int, factor: float, splits: tuple, disp,
             preentrenado: bool = True, conservar_pesos: bool = True, log=print) -> dict:
    from ultralytics import YOLO

    spec = R.MODELOS[modelo]
    lote = spec["lote"]
    we = R.calentamiento_epocas(epocas, n_train())
    info = {"lr_inicial": LR_POR_DEFECTO * factor, "lr_por_defecto": LR_POR_DEFECTO, "optimizador": "AdamW",
            "lote": lote, "acumulacion": R.acumulacion(modelo), "calentamiento_epocas": we}
    log(f"  {modelo}: {info}")
    yolo = YOLO(f"{spec['variante']}.pt" if preentrenado else f"{spec['variante']}.yaml")
    t0 = time.time()
    yolo.train(data=str(DATOS / "yolo" / "data.yaml"), imgsz=1280, epochs=epocas, batch=lote, nbs=R.LOTE_EFECTIVO,
               seed=semilla, deterministic=True, workers=2, device=0 if disp.type == "cuda" else "cpu",
               project=str(carpeta.resolve()), name="ultra", exist_ok=True, plots=False, verbose=False,
               amp=disp.type == "cuda", pretrained=preentrenado, patience=10 * epocas + 10, val=True,
               optimizer="AdamW", lr0=LR_POR_DEFECTO * factor, momentum=0.9, warmup_bias_lr=0.0,
               warmup_epochs=we, cos_lr=True, lrf=R.LR_FINAL, save_period=1, **AUMENTACION)
    t_entrenar = time.time() - t0
    memoria = pico_gb()
    lote_real = yolo.trainer.batch_size
    if lote_real != lote:  # ultralytics 8.4 baja el lote solo si hay OOM en la primera época
        raise RuntimeError(f"ultralytics redujo el lote de {lote} a {lote_real} por falta de memoria")
    pesos = carpeta / "ultra" / "weights"
    filas = leer_results(carpeta / "ultra" / "results.csv")
    mejor = max(filas, key=lambda f: (f["metrics/mAP50-95(M)"], -f["epoch"]))  # empate: la época más temprana
    ep = int(mejor["epoch"])
    ckpt = pesos / f"epoch{ep - 1}.pt"
    t1 = time.time()
    elegido = YOLO(str(ckpt))
    for split in splits:
        guardar_preds(carpeta, split, predecir(elegido, split, disp))
    t_predecir = time.time() - t1
    params = sum(p.numel() for p in elegido.model.parameters())
    if conservar_pesos:
        shutil.copy2(ckpt, carpeta / "mejor.pt")
    shutil.rmtree(pesos)
    hist = [{"epoca": int(f["epoch"]), "perdida": sum(v for k, v in f.items() if k.startswith("train/")),
             "val_AP": f["metrics/mAP50-95(M)"], "val_AP50": f["metrics/mAP50(M)"],
             "val_AP_caja": f.get("metrics/mAP50-95(B)")} for f in filas]
    return {**info, **memoria, "t_entrenar_s": t_entrenar, "t_predecir_s": t_predecir, "mejor_epoca": ep,
            "parametros": params, "historial": hist}
