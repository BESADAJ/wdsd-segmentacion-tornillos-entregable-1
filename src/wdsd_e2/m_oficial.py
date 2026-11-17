"""Trabajador de los repositorios oficiales (entorno wdsd_yolo9): YOLOv7-seg (u7, carpeta seg/) y YOLOv9-seg (GELAN-C).

Mismos cambios declarados en results/e2/prueba_humo/repos_oficiales/README.md: nbs = 2 en segment/train.py y, en
YOLOv9, max(target_scores.sum(), 1) en utils/segment/loss_tal.py (los dos se verifican antes de entrenar), y parches de NumPy. El hyp se genera desde el hyp.scratch-high.yaml de cada repo:
- lr0 = 0.01 x factor (SGD, el optimizador por defecto de los dos repos), lrf = 0.01, coseno (--cos-lr);
- warmup_epochs = calentamiento común del E2 (el repo además impone un mínimo de 100 iteraciones);
- única aumentación: fliplr 0.5 (mosaic, mixup, copy_paste, HSV, escala, traslación en 0).
Mejor época: metrics/mAP_0.5:0.95(M) de results.csv (época desde 0; la última fila, con época >= E, es la evaluación
final del best.pt del repo y se ignora). Predicciones: segment/val.py --save-json con conf 0.05, NMS 0.6 y 100
detecciones, como el E1; las máscaras salen en RLE a resolución original.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import registro as R
from .comun import DATOS, guardar_preds, imagenes, n_train

AUMENTACION_CERO = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud",
                    "mosaic", "mixup", "copy_paste")
LR_POR_DEFECTO = 0.01


def verificar_repo(repo: Path, spec: dict):
    train = repo / "segment" / "train.py"
    if not train.exists():
        raise FileNotFoundError(f"No encuentro {train}. --repo debe ser la carpeta yolov9/ o yolov7u7/seg/.")
    if not re.search(r"^\s*nbs\s*=\s*2\b", train.read_text(encoding="utf8"), re.M):
        raise RuntimeError(f"{train} no tiene el parche nbs = 2 (repos_oficiales/README.md, cambio 1).")
    perdida = repo / "utils" / "segment" / "loss_tal.py"  # solo YOLOv9 (YOLOv7 u7 usa utils/segment/loss.py)
    if perdida.exists() and "max(target_scores.sum(), 1)" not in perdida.read_text(encoding="utf8"):
        raise RuntimeError(f"{perdida} no tiene el parche max(target_scores.sum(), 1) (repos_oficiales/README.md, cambio 5).")
    if not (repo / spec["pesos"]).exists():
        raise FileNotFoundError(f"Faltan los pesos COCO {repo / spec['pesos']}")


def escribir_hyp(repo: Path, destino: Path, factor: float, warmup_epocas: float):
    import yaml
    hyp = yaml.safe_load((repo / "data" / "hyps" / "hyp.scratch-high.yaml").read_text(encoding="utf8"))
    hyp.update(lr0=LR_POR_DEFECTO * factor, lrf=R.LR_FINAL, warmup_epochs=round(warmup_epocas, 4), fliplr=0.5)
    for k in AUMENTACION_CERO:
        if k in hyp:
            hyp[k] = 0.0
    destino.write_text(yaml.safe_dump(hyp, sort_keys=False), encoding="utf8")
    return hyp


def escribir_data(destino: Path, val: str):
    import yaml
    d = {"path": (DATOS / "yolo").resolve().as_posix(), "train": "images/train", "val": val, "nc": 1,
         "names": {0: "tornillo"}}
    destino.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf8")


def _correr(cmd: list[str], cwd: Path, log_path: Path, log=print) -> str:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    texto = []
    with log_path.open("a", encoding="utf8") as f, subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8",
            errors="replace") as p:
        f.write("$ " + " ".join(cmd) + "\n")
        for linea in p.stdout:
            f.write(linea); texto.append(linea)
            if " all " in linea or "Error" in linea or "Traceback" in linea:
                log("    " + linea.rstrip())
        p.wait()
    if p.returncode:
        raise RuntimeError(f"{cmd[1]} terminó con código {p.returncode}; ver {log_path}")
    return "".join(texto)


def leer_results(ruta: Path, epocas: int) -> list[dict]:
    with ruta.open(encoding="utf8") as f:
        # la última fila (evaluación final del best.pt del repo) tiene menos columnas que el encabezado: se descarta
        filas = [{k.strip(): float(v) for k, v in fila.items() if k is not None and v not in (None, "")}
                 for fila in csv.DictReader(f)]
    return [f for f in filas if f["epoch"] < epocas and "metrics/mAP_0.5:0.95(M)" in f]


def predecir(repo: Path, pesos: Path, split: str, carpeta: Path, disp, log=print) -> list[dict]:
    val = {"val": "images/val", "test": "images/test", "train93": (DATOS / "yolo" / "train93.txt").resolve().as_posix()}[split]
    data = carpeta / f"data_{split}.yaml"
    escribir_data(data, val)
    salida = carpeta / f"pred_{split}"
    _correr([sys.executable, "segment/val.py", "--data", str(data), "--weights", str(pesos), "--img", "1280",
             "--batch-size", "1", "--task", "val", "--conf-thres", "0.05", "--iou-thres", "0.6", "--max-det", "100",
             "--save-json", "--device", "0" if disp.type == "cuda" else "cpu", "--project", str(carpeta),
             "--name", salida.name, "--exist-ok"], repo, carpeta / "val.log", log)
    archivo = salida / f"{pesos.stem}_predictions.json"
    crudo = json.loads(archivo.read_text()) if archivo.exists() else []  # sin detecciones el repo no escribe el archivo
    ids = {Path(im["file_name"]).stem: im for im in imagenes(split)}
    preds = []
    for p in crudo:
        im = ids[str(p["image_id"])]
        seg = p["segmentation"]
        if list(seg["size"]) != [im["height"], im["width"]]:
            raise ValueError(f"Máscara de tamaño {seg['size']}, esperaba {[im['height'], im['width']]}")
        preds.append({"image_id": im["id"], "category_id": 1, "segmentation": seg, "score": float(p["score"])})
    from pycocotools import mask as mask_utils
    preds = [p for p in preds if mask_utils.area(p["segmentation"]) > 0]
    shutil.rmtree(salida)
    return preds


def entrenar(modelo: str, carpeta: Path, semilla: int, epocas: int, factor: float, splits: tuple, disp,
             repo: Path, preentrenado: bool = True, conservar_pesos: bool = True, log=print) -> dict:
    import torch
    spec = R.MODELOS[modelo]
    repo = Path(repo).resolve()
    verificar_repo(repo, spec)
    we = R.calentamiento_epocas(epocas, n_train())
    hyp = escribir_hyp(repo, carpeta / "hyp.yaml", factor, we)
    escribir_data(carpeta / "data_train.yaml", "images/val")
    info = {"lr_inicial": hyp["lr0"], "lr_por_defecto": LR_POR_DEFECTO, "optimizador": "SGD", "lote": spec["lote"],
            "acumulacion": R.acumulacion(modelo), "calentamiento_epocas": hyp["warmup_epochs"]}
    log(f"  {modelo}: {info}")
    t0 = time.time()
    texto = _correr([sys.executable, "segment/train.py", "--data", str(carpeta / "data_train.yaml"),
                     "--cfg", spec["cfg"], "--weights", spec["pesos"] if preentrenado else "",
                     "--hyp", str(carpeta / "hyp.yaml"), "--img", "1280", "--batch-size", str(spec["lote"]),
                     "--epochs", str(epocas), "--device", "0" if disp.type == "cuda" else "cpu", "--workers", "0",
                     "--project", str(carpeta), "--name", "oficial", "--exist-ok", "--seed", str(semilla),
                     "--cos-lr", "--save-period", "1", "--noplots", "--patience", str(10 * epocas + 10)],
                    repo, carpeta / "train.log", log)
    t_entrenar = time.time() - t0
    # GPU_mem del log = memoria reservada por PyTorch en GB de 10^9 bytes (repos_oficiales/README.md)
    mem = [float(x) for x in re.findall(r"\s(\d+(?:\.\d+)?)G\s", texto)]
    total = 6.44  # GB (10^9) de las tarjetas de 6 GB; no se consulta a CUDA para no abrir un contexto en este proceso
    pico = max(mem) if mem else None
    filas = leer_results(carpeta / "oficial" / "results.csv", epocas)
    mejor = max(filas, key=lambda f: (f["metrics/mAP_0.5:0.95(M)"], -f["epoch"]))
    ep0 = int(mejor["epoch"])
    ckpt = carpeta / "oficial" / "weights" / spec["patron_ckpt"].format(ep0)
    elegido = carpeta / "mejor.pt"
    shutil.copy2(ckpt, elegido)
    shutil.rmtree(carpeta / "oficial" / "weights")
    t1 = time.time()
    for split in splits:
        guardar_preds(carpeta, split, predecir(repo, elegido, split, carpeta, disp, log))
    t_predecir = time.time() - t1
    m = re.search(r"summary: [^\n]*?([\d,]+) parameters", texto)  # resumen del modelo que imprime train.py
    params = int(m.group(1).replace(",", "")) if m else None
    if not conservar_pesos:
        elegido.unlink()
    hist = [{"epoca": int(f["epoch"]) + 1, "perdida": sum(v for k, v in f.items() if k.startswith("train/")),
             "val_AP": f["metrics/mAP_0.5:0.95(M)"], "val_AP50": f["metrics/mAP_0.5(M)"],
             "val_AP_caja": f["metrics/mAP_0.5:0.95(B)"]} for f in filas]
    return {**info, "gpu_mem_log_pico_gb": pico, "desborde_vram": bool(pico and pico > total),
            "t_entrenar_s": t_entrenar, "t_predecir_s": t_predecir, "mejor_epoca": ep0 + 1, "parametros": params,
            "historial": hist}
