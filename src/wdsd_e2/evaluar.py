"""Evaluación del E2 con las métricas del E1 (src/wdsd_inst/metricas.py), sin cambios.

- búsqueda: solo val. Umbral de score elegido en val (el que maximiza el F1 medio por imagen, como en el E1) y
  F1 de instancia promediado por grupo en val, que es el criterio para elegir la configuración.
- final: evaluar_corrida del E1 (val y test; test_por_imagen.csv y metricas.json) y, además, F1 en la submuestra fija
  de 93 imágenes de train con el mismo umbral, para la brecha de sobreajuste (δg, RQ4).
"""
from __future__ import annotations

import json
from pathlib import Path

from wdsd_inst.metricas import ap_mascara, cargar_gt, elegir_umbral, evaluar_corrida, pq_por_imagen

from .comun import ruta_json


def _cargar(carpeta: Path, split: str) -> list[dict]:
    return json.loads((carpeta / f"preds_{split}.json").read_text())


def busqueda(carpeta: Path) -> dict:
    gt = cargar_gt(ruta_json("val"))
    pv = _cargar(carpeta, "val")
    umbral, curva = elegir_umbral(gt, pv)
    curva.to_csv(carpeta / "umbral_val.csv", index=False)
    tv = pq_por_imagen(gt, pv, umbral)
    tv.to_csv(carpeta / "val_por_imagen.csv", index=False)
    m = {"umbral_score": umbral,
         "val": {**ap_mascara(gt, pv), "F1_img": tv.f1.mean(), "F1_grupo": tv.groupby("grupo").f1.mean().mean(),
                 "PQ_grupo": tv.groupby("grupo").pq.mean().mean()}}
    (carpeta / "metricas.json").write_text(json.dumps(m, indent=1), encoding="utf8")
    return m


def final(carpeta: Path) -> dict:
    m = evaluar_corrida(carpeta, ruta_json("val"), ruta_json("test"))
    gt_val = cargar_gt(ruta_json("val"))
    pq_por_imagen(gt_val, _cargar(carpeta, "val"), m["umbral_score"]).to_csv(carpeta / "val_por_imagen.csv", index=False)
    if (carpeta / "preds_train93.json").exists():
        gt = cargar_gt(ruta_json("train93"))
        tr = pq_por_imagen(gt, _cargar(carpeta, "train93"), m["umbral_score"])
        tr.to_csv(carpeta / "train93_por_imagen.csv", index=False)
        m["train93"] = {"F1_img": tr.f1.mean(), "F1_grupo": tr.groupby("grupo").f1.mean().mean()}
        m["brecha_F1_img_train93_menos_val"] = m["train93"]["F1_img"] - m["val"]["F1_img"]
        (carpeta / "metricas.json").write_text(json.dumps(m, indent=1), encoding="utf8")
    return m
