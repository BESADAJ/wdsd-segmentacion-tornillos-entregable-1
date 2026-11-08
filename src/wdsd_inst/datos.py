"""Datos para segmentación de instancias: COCO reagrupado con la partición 744/93/108 y formato YOLO.

Fuente de verdad:
- Anotaciones COCO oficiales: <raiz>/Annotations/Object Detection & Instance Segmentation COCO-format/*.json
- Partición agrupada congelada: <proyecto>/results/eda/particion.csv (columnas archivo, split_oficial, split, grupo)
- Imágenes: <raiz>/{train,val,test}/<archivo> (o carpetas WDSD-*/WDSD, igual que el código semántico)

La clave de unión es (split_oficial, archivo): un mismo nombre de archivo puede repetirse entre
particiones oficiales, por eso el nombre nuevo es "<split_oficial>__<archivo>".
"""
from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

COCO_DIR = "Annotations/Object Detection & Instance Segmentation COCO-format"
COCO_FILES = {"train": "Training_set.json", "val": "Validation_set.json", "test": "Test_set.json"}
SPLITS = ("train", "val", "test")


def raices_imagenes(data_root: Path) -> list[Path]:
    if (data_root / "train").is_dir():
        return [data_root]
    roots = sorted(data_root.glob("WDSD-*/WDSD")) + sorted(data_root.glob("WDSD"))
    if not roots:
        raise FileNotFoundError(f"No encuentro train/ ni WDSD-*/WDSD en {data_root}. Usa --data-root.")
    return roots


def archivo_coco(data_root: Path, nombre: str) -> Path:
    """Busca un JSON COCO en todas las raíces: con la descarga en dos zips, cada uno trae solo algunos."""
    for r in raices_imagenes(data_root):
        if (r / COCO_DIR / nombre).is_file():
            return r / COCO_DIR / nombre
    raise FileNotFoundError(f"No encuentro {COCO_DIR}/{nombre} bajo {data_root}")


def preparar(data_root: Path, particion_csv: Path, salida: Path, yolo: bool = True) -> dict:
    """Escribe salida/coco/{train,val,test}.json y, si yolo, salida/yolo/{images,labels}/<split>/ + data.yaml."""
    part = pd.read_csv(particion_csv)
    info = {(r.split_oficial, r.archivo): r for r in part.itertuples()}
    rutas = {}
    for root in raices_imagenes(data_root):
        for s in SPLITS:
            for p in (root / s).glob("*.png"):
                rutas[(s, p.name)] = p

    nuevo = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "tornillo", "supercategory": "tornillo"}]}
             for s in SPLITS}
    vistos = set()
    for s_of, nombre in COCO_FILES.items():
        d = json.loads(archivo_coco(data_root, nombre).read_text(encoding="utf8"))
        por_img = defaultdict(list)
        for a in d["annotations"]:
            por_img[a["image_id"]].append(a)
        for im in d["images"]:
            clave = (s_of, im["file_name"])
            r = info.get(clave)
            if r is None:
                raise KeyError(f"{clave} no está en {particion_csv}")
            if clave not in rutas:
                raise FileNotFoundError(f"No encuentro la imagen {clave} bajo {data_root}")
            vistos.add(clave)
            dst = nuevo[r.split]
            img_id = len(dst["images"]) + 1
            dst["images"].append({"id": img_id, "file_name": f"{s_of}__{im['file_name']}", "width": im["width"],
                                  "height": im["height"], "ruta": str(rutas[clave]), "grupo": int(r.grupo),
                                  "archivo": im["file_name"], "split_oficial": s_of})
            for a in por_img[im["id"]]:
                dst["annotations"].append({"id": len(dst["annotations"]) + 1, "image_id": img_id, "category_id": 1,
                                           "iscrowd": 0, "area": a["area"], "bbox": a["bbox"],
                                           "segmentation": a["segmentation"]})
    faltan = set(info) - vistos
    if faltan:
        raise KeyError(f"{len(faltan)} imágenes de la partición no tienen anotación COCO, p. ej. {list(faltan)[:3]}")

    (salida / "coco").mkdir(parents=True, exist_ok=True)
    resumen = {}
    for s in SPLITS:
        (salida / "coco" / f"{s}.json").write_text(json.dumps(nuevo[s]), encoding="utf8")
        resumen[s] = {"imagenes": len(nuevo[s]["images"]), "instancias": len(nuevo[s]["annotations"]),
                      "grupos": len({i["grupo"] for i in nuevo[s]["images"]})}
    if yolo:
        _escribir_yolo(nuevo, salida / "yolo")
    (salida / "resumen_preparacion.json").write_text(json.dumps(resumen, indent=1), encoding="utf8")
    return resumen


def _enlazar(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)  # enlace duro: no ocupa espacio extra (mismo disco)
    except OSError:
        shutil.copy2(src, dst)


def _escribir_yolo(nuevo: dict, raiz: Path):
    """Formato de ultralytics: una línea por instancia, 'clase x1 y1 x2 y2 ...' normalizado a [0, 1].
    Dos anotaciones de WDSD tienen más de un polígono: se usa el de mayor número de vértices."""
    for s in SPLITS:
        (raiz / "images" / s).mkdir(parents=True, exist_ok=True)
        (raiz / "labels" / s).mkdir(parents=True, exist_ok=True)
        por_img = defaultdict(list)
        for a in nuevo[s]["annotations"]:
            por_img[a["image_id"]].append(a)
        for im in nuevo[s]["images"]:
            nombre = Path(im["file_name"]).stem
            _enlazar(Path(im["ruta"]), raiz / "images" / s / f"{nombre}.png")
            w, h = im["width"], im["height"]
            lineas = []
            for a in por_img[im["id"]]:
                poli = max(a["segmentation"], key=len)
                xy = [f"{min(max(v / (w if k % 2 == 0 else h), 0), 1):.6f}" for k, v in enumerate(poli)]
                lineas.append("0 " + " ".join(xy))
            (raiz / "labels" / s / f"{nombre}.txt").write_text("\n".join(lineas) + "\n", encoding="utf8")
    (raiz / "data.yaml").write_text(
        f"path: {raiz.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: tornillo\n",
        encoding="utf8")
