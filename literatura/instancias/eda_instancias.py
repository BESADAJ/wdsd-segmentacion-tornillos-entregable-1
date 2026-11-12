"""Análisis exploratorio auxiliar de instancias de WDSD.

Lee las anotaciones COCO y la partición agrupada, y escribe tablas CSV y un
resumen JSON. No contiene el EDA completo ni genera las tablas del informe final.

La función rasterizar usa PIL.ImageDraw.polygon con contorno. Esa máscara puede
tener más píxeles que el campo `area` de COCO; no es la referencia del benchmark.
El análisis final rasteriza por centros de píxel con
`src/wdsd_e1/instancias.py::etiqueta_coco`.

Uso desde la raíz del repositorio:
    python literatura/instancias/eda_instancias.py --raiz . --salida results/eda_instancias

Lee los tres JSON COCO bajo `Annotations/Object Detection & Instance
Segmentation COCO-format` y `results/eda/particion.csv`.



"""

from __future__ import annotations #Para indicar el type

import argparse 
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

H, W = 720, 1280
COCO = "Annotations/Object Detection & Instance Segmentation COCO-format"
COCO_FILES = {"train": "Training_set.json", "val": "Validation_set.json", "test": "Test_set.json"}
ORDEN = ["train", "val", "test"]


def rasterizar(seg: list[list[float]]) -> np.ndarray:
    """Máscara exploratoria con Pillow; incluye el contorno del polígono.

    No usar como verdad de referencia: el EDA final usa N.etiqueta_coco.
    """
    lienzo = Image.new("L", (W, H), 0) #Creamos una objeto tipo Image que es una imagen en blanco
    d = ImageDraw.Draw(lienzo) #Este objeto dibuja sobre el lienzo
    for poli in seg: #Iteramos sobre la lista de poligonos "seg" 
        d.polygon(poli, fill=1, outline=1) #dibuja y rellena un polígono geométrico sobre el lienzo digital
    return np.asarray(lienzo, dtype=bool) #Convertimos a bool que es estado logio de donde hay pixel y donde no 


def dilatar(m: np.ndarray) -> np.ndarray:
    """Dilatación de 1 px con vecindad 8 (sin scipy)."""
    p = np.pad(m, 1)
    out = np.zeros_like(m)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= p[dy:dy + H, dx:dx + W]
    return out


def cargar(raiz: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = pd.read_csv(raiz / "results/eda/particion.csv")
    clave = {(r.split_oficial, r.archivo): r for r in part.itertuples()}
    imgs, inst = [], []
    for split_of, nombre in COCO_FILES.items():
        d = json.loads((raiz / COCO / nombre).read_text(encoding="utf8"))
        por_img = defaultdict(list)
        for a in d["annotations"]:
            por_img[a["image_id"]].append(a)
        for im in d["images"]:
            r = clave[(split_of, im["file_name"])]  # falla si una imagen no está en la partición
            anns = por_img[im["id"]]
            mascaras = [rasterizar(a["segmentation"]) for a in anns]
            # contacto: la máscara dilatada 1 px de una instancia toca o pisa otra
            toca = np.zeros(len(anns), dtype=bool)
            solape = np.zeros(len(anns), dtype=bool)
            for i in range(len(anns)):
                di = dilatar(mascaras[i])
                for j in range(i + 1, len(anns)):
                    if (mascaras[i] & mascaras[j]).any():
                        solape[i] = solape[j] = True
                    if (di & mascaras[j]).any():
                        toca[i] = toca[j] = True
            imgs.append({"archivo": im["file_name"], "split_oficial": split_of, "split": r.split,
                         "grupo": r.grupo, "familia": r.familia, "n_inst": len(anns),
                         "n_tocan": int(toca.sum()), "n_solapan": int(solape.sum())})
            for a, m, t, s in zip(anns, mascaras, toca, solape):
                x, y, w, h = a["bbox"]
                px = int(m.sum())
                inst.append({"archivo": im["file_name"], "split": r.split, "grupo": r.grupo,
                             "area_coco": a["area"], "px_mascara": px, "bbox_w": w, "bbox_h": h,
                             "lado_bbox": max(w, h), "diam_equiv": 2 * np.sqrt(px / np.pi),
                             "n_poligonos": len(a["segmentation"]), "iscrowd": a.get("iscrowd", 0),
                             "toca_otra": bool(t), "solapa_otra": bool(s),
                             "cx": x + w / 2, "cy": y + h / 2})
    return pd.DataFrame(imgs), pd.DataFrame(inst)


def tamano_coco(area: pd.Series) -> pd.Series:
    """Cortes de COCO para AP_S / AP_M / AP_L: 32² y 96² píxeles."""
    return pd.cut(area, [-1, 32 ** 2, 96 ** 2, np.inf], labels=["pequeño", "mediano", "grande"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", type=Path, default=Path("."))
    ap.add_argument("--salida", type=Path, default=Path("results/eda_instancias"))
    a = ap.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    imgs, inst = cargar(a.raiz)
    imgs.to_csv(a.salida / "por_imagen.csv", index=False)
    inst.to_csv(a.salida / "por_instancia.csv", index=False)

    inst["tamano_coco"] = tamano_coco(inst.area_coco)
    filas = []
    for s in ORDEN + ["total"]:
        im = imgs if s == "total" else imgs[imgs.split == s]
        ins = inst if s == "total" else inst[inst.split == s]
        filas.append({
            "split": s, "imagenes": len(im), "grupos": im.grupo.nunique(), "instancias": len(ins),
            "inst_por_img_media": im.n_inst.mean(), "inst_por_img_mediana": im.n_inst.median(),
            "inst_por_img_max": im.n_inst.max(), "pct_img_sin_tornillo": 100 * (im.n_inst == 0).mean(),
            "pct_inst_que_tocan": 100 * ins.toca_otra.mean(), "pct_inst_que_solapan": 100 * ins.solapa_otra.mean(),
            "pct_img_con_contacto": 100 * (im.n_tocan > 0).mean(),
            "area_mediana_px": ins.area_coco.median(), "diam_equiv_mediano_px": ins.diam_equiv.median(),
            "lado_bbox_mediano_px": ins.lado_bbox.median(),
            "pct_pequeno_coco": 100 * (ins.tamano_coco == "pequeño").mean(),
            "pct_mediano_coco": 100 * (ins.tamano_coco == "mediano").mean(),
            "pct_grande_coco": 100 * (ins.tamano_coco == "grande").mean(),
        })
    resumen = pd.DataFrame(filas)
    resumen.to_csv(a.salida / "resumen_por_split.csv", index=False)

    # Sección 8: qué tamaño tiene el tornillo después de reescalar la imagen
    escalas = {"nativa 1280": 1.0, "YOLO imgsz 640": 0.5, "YOLO imgsz 960": 0.75, "Mask R-CNN lado corto 800": 800 / 720}
    res = []
    for nombre, f in escalas.items():
        d = inst.diam_equiv * f
        l = inst.lado_bbox * f
        res.append({"escala": nombre, "factor": round(f, 3), "diam_mediano_px": d.median(),
                    "p10_diam_px": d.quantile(0.10), "pct_diam_<8px": 100 * (d < 8).mean(),
                    "lado_bbox_mediano_px": l.median(), "pct_lado_bbox_<16px": 100 * (l < 16).mean(),
                    "pct_lado_bbox_<32px": 100 * (l < 32).mean()})
    pd.DataFrame(res).to_csv(a.salida / "efecto_escala.csv", index=False)

    # Por grupo en test: cuántas instancias aporta cada unidad de independencia
    pg = (imgs[imgs.split == "test"].groupby("grupo")
          .agg(imagenes=("n_inst", "size"), instancias=("n_inst", "sum"), familia=("familia", "first"))
          .sort_values("instancias", ascending=False))
    pg.to_csv(a.salida / "test_por_grupo.csv")

    json.dump({"resumen": resumen.round(3).to_dict(orient="records"),
               "n_poligonos_>1": int((inst.n_poligonos > 1).sum()),
               "iscrowd": int(inst.iscrowd.sum()),
               "acuerdo_area_coco_vs_raster_mediana": float((inst.px_mascara / inst.area_coco).median())},
              open(a.salida / "resumen.json", "w"), ensure_ascii=False, indent=1)
    print(resumen.round(2).T.to_string())
    print(pd.DataFrame(res).round(2).to_string())
    print(pg.to_string())


if __name__ == "__main__":
    main()
