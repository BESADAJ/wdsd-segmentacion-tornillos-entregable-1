"""Mask R-CNN (torchvision, ResNet-50-FPN preentrenada en COCO) para tornillos.

Cambios respecto a la configuración por defecto, justificados por el EDA de instancias:
- Sin reescalado: min_size=720, max_size=1280 (por defecto torchvision lleva el lado corto a 800).
- Anclas de 8, 16, 32, 64 y 128 px (por defecto 32–512): el lado mediano de la caja de un tornillo es 12 px
  y el 97 % mide menos de 32 px. El número de anclas por posición (3) no cambia, así que la cabeza RPN
  preentrenada se carga sin problema.
- Cabezas de caja y de máscara reemplazadas para 2 clases (fondo + tornillo).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor

from .metricas import a_rle, ap_mascara, cargar_gt


class DatasetCOCO(torch.utils.data.Dataset):
    def __init__(self, ruta_json: Path, voltear: bool = False):
        d = json.loads(Path(ruta_json).read_text(encoding="utf8"))
        self.imgs = d["images"]
        self.anns = {i["id"]: [] for i in self.imgs}
        for a in d["annotations"]:
            self.anns[a["image_id"]].append(a)
        self.voltear = voltear

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, k):
        info = self.imgs[k]
        img = pil_to_tensor(Image.open(info["ruta"]).convert("RGB")).float() / 255.0
        h, w = info["height"], info["width"]
        masks, boxes = [], []
        for a in self.anns[info["id"]]:
            # misma rasterización que la evaluación (pycocotools, área = campo "area" de COCO);
            # ImageDraw.polygon con borde engorda cada tornillo ~19 px (EDA de instancias)
            m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], h, w))).astype(np.uint8)
            ys, xs = np.nonzero(m)
            masks.append(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        masks = torch.as_tensor(np.stack(masks) if masks else np.zeros((0, h, w), np.uint8))
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.voltear and random.random() < 0.5:
            img, masks = img.flip(-1), masks.flip(-1)
            boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64), "masks": masks,
                  "image_id": info["id"]}
        return img, target


def _juntar(lote):
    return tuple(zip(*lote))


def crear_modelo(preentrenado: bool = True) -> torch.nn.Module:
    anclas = AnchorGenerator(((8,), (16,), (32,), (64,), (128,)), ((0.5, 1.0, 2.0),) * 5)
    pesos = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if preentrenado else None
    m = maskrcnn_resnet50_fpn(weights=pesos, weights_backbone=None, num_classes=None if preentrenado else 2,
                              min_size=720, max_size=1280, rpn_anchor_generator=anclas,
                              box_detections_per_img=100)
    if preentrenado:
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
        m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    return m


@torch.no_grad()
def predecir(modelo, ruta_json: Path, disp, umbral_score: float = 0.05) -> list[dict]:
    modelo.eval()
    ds = DatasetCOCO(ruta_json)
    salida = []
    for k in range(len(ds)):
        img, t = ds[k]
        with torch.autocast(device_type=disp.type, enabled=disp.type == "cuda"):
            out = modelo([img.to(disp)])[0]
        sc = out["scores"].float().cpu().numpy()
        ms = (out["masks"][:, 0] > 0.5).cpu().numpy()
        for s, m in zip(sc, ms):
            if s >= umbral_score and m.any():
                salida.append({"image_id": int(t["image_id"]), "category_id": 1, "segmentation": a_rle(m),
                               "score": float(s)})
    return salida


def entrenar(datos: Path, carpeta: Path, semilla: int, epocas: int, disp, lote: int = 2, lr: float | None = None,
             preentrenado: bool = True, max_iter: int | None = None, log=print) -> dict:
    """Entrena, selecciona la mejor época por AP de máscara en val y deja preds_val/preds_test.json."""
    carpeta.mkdir(parents=True, exist_ok=True)
    lr = lr if lr is not None else 0.0025 * lote  # escala lineal: 0.02 para lote 8x2 en la receta de torchvision
    g = torch.Generator().manual_seed(semilla)
    dl = torch.utils.data.DataLoader(DatasetCOCO(datos / "coco/train.json", voltear=True), batch_size=lote,
                                     shuffle=True, num_workers=0, collate_fn=_juntar, generator=g)
    modelo = crear_modelo(preentrenado).to(disp)
    params = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    pasos = epocas * (len(dl) if max_iter is None else min(max_iter, len(dl)))
    calentamiento = min(500, max(1, pasos // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / calentamiento if i < calentamiento
        else 0.5 * (1 + math.cos(math.pi * (i - calentamiento) / max(1, pasos - calentamiento))))
    scaler = torch.amp.GradScaler(enabled=disp.type == "cuda")
    gt_val = cargar_gt(datos / "coco/val.json")
    historial, mejor = [], -1.0
    if disp.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(1, epocas + 1):
        modelo.train()
        t0, suma, n = time.time(), 0.0, 0
        for it, (imgs, tgts) in enumerate(dl):
            if max_iter is not None and it >= max_iter:
                break
            imgs = [i.to(disp) for i in imgs]
            tgts = [{k: (v.to(disp) if torch.is_tensor(v) else v) for k, v in t.items()} for t in tgts]
            with torch.autocast(device_type=disp.type, enabled=disp.type == "cuda"):
                perdidas = modelo(imgs, tgts)
                perdida = sum(perdidas.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(perdida).backward()
            scaler.step(opt); scaler.update(); sched.step()
            suma += float(perdida); n += 1
        t_train = time.time() - t0
        preds = predecir(modelo, datos / "coco/val.json", disp)
        ap = ap_mascara(gt_val, preds)
        fila = {"epoca": ep, "perdida": suma / max(n, 1), "s_train": t_train, "s_total": time.time() - t0,
                **{f"val_{k}": v for k, v in ap.items()}}
        historial.append(fila)
        log(f"  maskrcnn s{semilla} época {ep}/{epocas}: pérdida {fila['perdida']:.3f}, val AP {ap['AP']:.3f}, "
            f"AP50 {ap['AP50']:.3f}, {fila['s_total']:.0f} s")
        if ap["AP"] > mejor:
            mejor = ap["AP"]
            torch.save(modelo.state_dict(), carpeta / "mejor.pt")
    modelo.load_state_dict(torch.load(carpeta / "mejor.pt", map_location=disp))
    (carpeta / "preds_val.json").write_text(json.dumps(predecir(modelo, datos / "coco/val.json", disp)))
    (carpeta / "preds_test.json").write_text(json.dumps(predecir(modelo, datos / "coco/test.json", disp)))
    pico = torch.cuda.max_memory_allocated() / 2**30 if disp.type == "cuda" else float("nan")
    return {"historial": historial, "mejor_val_AP": mejor, "gpu_pico_gb": pico,
            "parametros": sum(p.numel() for p in modelo.parameters())}
