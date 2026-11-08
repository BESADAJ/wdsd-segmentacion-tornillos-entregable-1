"""Métricas comunes para los tres modelos, a partir de predicciones en formato de resultados COCO.

Cada modelo escribe, para val y test, una lista de dicts:
    {"image_id": int, "category_id": 1, "segmentation": RLE, "score": float}
Así la evaluación es idéntica para Mask R-CNN, YOLOv8-seg y SOLOv2.

- AP de máscara (COCOeval, iouType="segm"): AP@[.5:.95], AP50, AP75 y AP_small. Sin umbral de score.
- Métrica primaria declarada en results/eda/declaracion_a_priori_instancias.json (EDA de instancias):
  F1 de instancia por imagen = 2TP / (2TP + FP + FN), con emparejamiento húngaro uno a uno por IoU de
  máscara y un par aceptado si IoU >= 0.5. Secundarias: SQ (IoU medio de los aciertos) y
  PQ = SQ x F1 (Panoptic Quality, Kirillov et al. 2019).
  Verdad de referencia: polígonos COCO rasterizados con pycocotools (área ~ campo "area" de COCO;
  mediana del cociente 1.00 en val), equivalente a la rasterización por centros de píxel del EDA.
  F1 y PQ necesitan un umbral de score: se elige en validación (el que maximiza el F1 medio por imagen)
  y se aplica sin cambios en test.
"""
from __future__ import annotations

import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

UMBRALES = np.round(np.arange(0.05, 0.96, 0.05), 2)


def a_rle(m: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def cargar_gt(ruta_json: Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(ruta_json))


def ap_mascara(gt: COCO, preds: list[dict]) -> dict:
    if not preds:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_small": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(preds)
        ev = COCOeval(gt, dt, iouType="segm")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]), "AP_small": float(s[3])}


def _gt_rles(gt: COCO, img_id: int) -> list:
    info = gt.imgs[img_id]
    h, w = info["height"], info["width"]
    salida = []
    for a in gt.imgToAnns[img_id]:
        seg = a["segmentation"]
        if isinstance(seg, dict) and not isinstance(seg.get("counts"), list):
            salida.append(seg)  # COCOeval ya lo convirtió a RLE en su lugar
            continue
        rles = mask_utils.frPyObjects(seg, h, w)
        salida.append(mask_utils.merge(rles) if isinstance(rles, list) else rles)
    return salida


def pq_por_imagen(gt: COCO, preds: list[dict], umbral: float) -> pd.DataFrame:
    por_img = defaultdict(list)
    for p in preds:
        if p["score"] >= umbral:
            por_img[p["image_id"]].append(p)
    filas = []
    for img_id in sorted(gt.imgs):
        info = gt.imgs[img_id]
        g = _gt_rles(gt, img_id)
        d = sorted(por_img[img_id], key=lambda p: -p["score"])
        d_rles = [p["segmentation"] for p in d]
        tp, suma_iou = 0, 0.0
        if g and d_rles:
            iou = mask_utils.iou(d_rles, g, [0] * len(g))  # (n_pred, n_gt)
            f, c = linear_sum_assignment(-iou)  # emparejamiento húngaro uno a uno por IoU
            ok = iou[f, c] >= 0.5
            tp = int(ok.sum())
            suma_iou = float(iou[f, c][ok].sum())
        fp, fn = len(d_rles) - tp, len(g) - tp
        den = tp + 0.5 * fp + 0.5 * fn
        # Dice semántico (unión de máscaras), para comparar con la etapa semántica
        if g or d_rles:
            ug = mask_utils.merge(g) if g else None
            ud = mask_utils.merge(d_rles) if d_rles else None
            a_g = float(mask_utils.area(ug)) if ug else 0.0
            a_d = float(mask_utils.area(ud)) if ud else 0.0
            inter = float(mask_utils.area(mask_utils.merge([ug, ud], intersect=True))) if (ug and ud) else 0.0
            dice = 2 * inter / (a_g + a_d) if (a_g + a_d) else 1.0
        else:
            dice = 1.0
        filas.append({"image_id": img_id, "archivo": info.get("file_name"), "grupo": info.get("grupo"),
                      "n_gt": len(g), "n_pred": len(d_rles), "tp": tp, "fp": fp, "fn": fn,
                      "f1": tp / den if den else 1.0,
                      "pq": suma_iou / den if den else 1.0,
                      "sq": suma_iou / tp if tp else 0.0,
                      "rq": tp / den if den else 1.0,
                      "dice_semantico": dice})
    return pd.DataFrame(filas)


def elegir_umbral(gt_val: COCO, preds_val: list[dict]) -> tuple[float, pd.DataFrame]:
    filas = []
    for u in UMBRALES:
        t = pq_por_imagen(gt_val, preds_val, u)
        filas.append({"umbral": u, "f1_medio": t.f1.mean(), "pq_medio": t.pq.mean()})
    curva = pd.DataFrame(filas)
    mejor = float(curva.loc[curva.f1_medio.idxmax(), "umbral"])
    return mejor, curva


def evaluar_corrida(carpeta: Path, gt_val_json: Path, gt_test_json: Path) -> dict:
    """Lee preds_val.json y preds_test.json de la carpeta; escribe test_por_imagen.csv y metricas.json."""
    gt_val, gt_test = cargar_gt(gt_val_json), cargar_gt(gt_test_json)
    pv = json.loads((carpeta / "preds_val.json").read_text())
    pt = json.loads((carpeta / "preds_test.json").read_text())
    umbral, curva = elegir_umbral(gt_val, pv)
    curva.to_csv(carpeta / "umbral_val.csv", index=False)
    tv = pq_por_imagen(gt_val, pv, umbral)
    tt = pq_por_imagen(gt_test, pt, umbral)
    tt.to_csv(carpeta / "test_por_imagen.csv", index=False)
    m = {"umbral_score": umbral,
         "val": {**ap_mascara(gt_val, pv), "F1_img": tv.f1.mean(), "PQ_img": tv.pq.mean(),
                 "F1_grupo": tv.groupby("grupo").f1.mean().mean(), "PQ_grupo": tv.groupby("grupo").pq.mean().mean()},
         "test": {**ap_mascara(gt_test, pt), "F1_img": tt.f1.mean(), "PQ_img": tt.pq.mean(), "SQ_img": tt.sq.mean(),
                  "F1_grupo": tt.groupby("grupo").f1.mean().mean(), "PQ_grupo": tt.groupby("grupo").pq.mean().mean(),
                  "dice_semantico_img": tt.dice_semantico.mean()}}
    (carpeta / "metricas.json").write_text(json.dumps(m, indent=1), encoding="utf8")
    return m
