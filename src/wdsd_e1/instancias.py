"""Medidas por instancia de WDSD a partir de las anotaciones COCO.

Cada tornillo anotado en COCO es una instancia. Este módulo las tabula, las
cruza con las componentes conexas de la máscara PNG y define la métrica de
instancia que se declara a priori (F1 por imagen con emparejamiento por IoU).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage import draw, measure

from . import eda as E

UMBRALES = (0.5, 0.75)


def etiqueta_coco(anns: list[dict]) -> np.ndarray:
    """Imagen de etiquetas: 0 fondo, k la k-ésima instancia de ``anns`` (desde 1).

    Rasteriza por centros de píxel: así el número de píxeles de cada instancia
    coincide con el campo ``area`` de COCO. ``ImageDraw.polygon`` añade el borde
    y engorda cada tornillo unos 19 px.
    """
    lab = np.zeros((E.H, E.W), np.int32)
    for k, a in enumerate(anns, start=1):
        for poli in a["segmentation"]:
            p = np.reshape(poli, (-1, 2))
            rr, cc = draw.polygon(p[:, 1], p[:, 0], (E.H, E.W))
            lab[rr, cc] = k
    return lab


def etiqueta_mascara(mascara: np.ndarray, area_min: int = 0) -> np.ndarray:
    """Instancias de una máscara binaria por componentes conexas (8-vecindad)."""
    lab = measure.label(mascara > 0, connectivity=2)
    if area_min:
        tam = np.bincount(lab.ravel())
        quitar = np.flatnonzero(tam < area_min)
        lab[np.isin(lab, quitar[quitar > 0])] = 0
        lab = measure.label(lab > 0, connectivity=2)
    return lab


def matriz_iou(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """IoU entre cada instancia de ``gt`` (filas) y de ``pred`` (columnas)."""
    ng, npr = int(gt.max()), int(pred.max())
    if ng == 0 or npr == 0:
        return np.zeros((ng, npr))
    conj = np.bincount(gt.ravel() * (npr + 1) + pred.ravel(), minlength=(ng + 1) * (npr + 1))
    inter = conj.reshape(ng + 1, npr + 1)[1:, 1:].astype(float)
    ag = np.bincount(gt.ravel(), minlength=ng + 1)[1:, None]
    ap = np.bincount(pred.ravel(), minlength=npr + 1)[None, 1:]
    return inter / (ag + ap - inter)


def metricas_instancia(gt: np.ndarray, pred: np.ndarray, umbral: float = 0.5) -> dict:
    """TP, FP, FN, F1, SQ y PQ de una imagen.

    Empareja uno a uno por máxima IoU (algoritmo húngaro) y acepta un par si su
    IoU es al menos ``umbral``. F1 es la métrica primaria declarada; SQ es la
    IoU media de los pares aceptados y PQ = SQ x F1 (calidad panóptica).
    Una imagen sin instancias reales ni predichas tiene F1 = 1.
    """
    iou = matriz_iou(gt, pred)
    ng, npr = iou.shape
    if ng and npr:
        f, c = linear_sum_assignment(-iou)
        ok = iou[f, c] >= umbral
        ious = iou[f, c][ok]
    else:
        ious = np.array([])
    tp = len(ious)
    fp, fn = npr - tp, ng - tp
    f1 = 1.0 if ng + npr == 0 else 2 * tp / (2 * tp + fp + fn)
    sq = float(ious.mean()) if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1, "sq": sq, "pq": sq * f1}


def _por_archivo(coco: dict) -> dict[str, list[dict]]:
    salida = defaultdict(list)
    for d in coco.values():
        nombre = {im["id"]: im["file_name"] for im in d["images"]}
        for a in d["annotations"]:
            salida[nombre[a["image_id"]]].append(a)
    return salida


por_archivo = _por_archivo
"""Anotaciones COCO agrupadas por nombre de archivo, uniendo los tres conjuntos oficiales."""


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return (a & b).sum() / max((a | b).sum(), 1)


def tablas_instancias(df: pd.DataFrame, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Una tabla por instancia COCO y otra por imagen, con caché en ``results/eda``.

    Por instancia: geometría declarada (bbox, área) y medida (polígono rasterizado),
    IoU con su componente del PNG (tal cual y corrigiendo 1 px en x), IoU consigo
    misma desplazada 1, 2 y 3 px, distancia al centro vecino más cercano y número
    de instancias que comparten su componente PNG.
    Por imagen: acuerdo COCO frente a PNG con la métrica de instancia.
    """
    c_ins, c_img = E.RESULTS / "instancias_coco.csv", E.RESULTS / "acuerdo_instancias.csv"
    if c_ins.exists() and c_img.exists() and not refresh:
        return pd.read_csv(c_ins), pd.read_csv(c_img)
    rutas = dict(zip(df.archivo, df.ruta_mask))
    split_of = dict(zip(df.archivo, df.split_oficial))
    filas_i, filas_m = [], []
    for archivo, anns in _por_archivo(E.carga_coco()).items():
        gt = etiqueta_coco(anns)
        png = etiqueta_mascara(np.asarray(Image.open(rutas[archivo])))
        cent = np.array([[a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2] for a in anns])
        nn = cKDTree(cent).query(cent, k=2)[0][:, 1] if len(anns) > 1 else np.full(1, np.nan)
        for k, a in enumerate(anns, start=1):
            m = gt == k
            ids = png[m]
            ids = ids[ids > 0]
            comp = np.bincount(ids).argmax() if ids.size else 0
            c = png == comp if comp else np.zeros_like(m)
            comparten = len(np.unique(gt[c & (gt > 0)])) if comp else 0
            pts = np.concatenate([np.reshape(p, (-1, 2)) for p in a["segmentation"]])
            x, y, w, h = a["bbox"]
            filas_i.append({
                "archivo": archivo, "split_oficial": split_of[archivo], "id": a["id"],
                "area_coco": a["area"], "px": int(m.sum()), "x": x, "y": y, "w": w, "h": h,
                "px_x0": pts[:, 0].min(), "px_x1": pts[:, 0].max(), "px_y0": pts[:, 1].min(), "px_y1": pts[:, 1].max(),
                "vertices": len(pts), "n_poligonos": len(a["segmentation"]),
                "cx": x + w / 2, "cy": y + h / 2, "dist_vecino": nn[k - 1],
                "iou_png": _iou(m, c), "iou_png_dx1": _iou(np.roll(m, 1, axis=1), c),
                "iou_auto1": _iou(np.roll(m, 1, axis=1), m), "iou_auto2": _iou(np.roll(m, 2, axis=1), m),
                "iou_auto3": _iou(np.roll(m, 3, axis=1), m), "instancias_en_componente": comparten,
            })
        fila = {"archivo": archivo, "n_coco": len(anns), "n_comp_png": int(png.max())}
        for u in UMBRALES:
            r = metricas_instancia(gt, png, u)
            fila.update({f"{k}_{u}": v for k, v in r.items()})
        filas_m.append(fila)
    ins, img = pd.DataFrame(filas_i), pd.DataFrame(filas_m)
    ins.to_csv(c_ins, index=False)
    img.to_csv(c_img, index=False)
    return ins, img


def cobertura_teselas(ins: pd.DataFrame, tam: int = 512, solape: int = 64) -> dict:
    """Instancias cortadas por una rejilla sin solape y cubiertas por ventanas con solape.

    Las ventanas de lado ``tam`` avanzan ``tam - solape`` y la última se ajusta
    al borde. Una instancia está cubierta si su caja cabe entera en una ventana.
    """
    def origenes(lado, paso):
        o = list(range(0, max(lado - tam, 0) + 1, paso))
        if o[-1] + tam < lado:
            o.append(lado - tam)
        return np.array(o)

    x0, y0 = ins.px_x0.to_numpy(), ins.px_y0.to_numpy()
    x1, y1 = ins.px_x1.to_numpy(), ins.px_y1.to_numpy()
    corte = (np.floor(x0 / tam) != np.floor(x1 / tam)) | (np.floor(y0 / tam) != np.floor(y1 / tam))
    ox, oy = origenes(E.W, tam - solape), origenes(E.H, tam - solape)
    cabe_x = ((ox[None] <= x0[:, None]) & (x1[:, None] <= ox[None] + tam)).any(1)
    cabe_y = ((oy[None] <= y0[:, None]) & (y1[:, None] <= oy[None] + tam)).any(1)
    return {"tam": tam, "solape": solape, "ventanas": len(ox) * len(oy),
            "cortadas_rejilla": float(corte.mean()), "cubiertas_ventana": float((cabe_x & cabe_y).mean())}
