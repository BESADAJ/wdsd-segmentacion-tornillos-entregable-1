"""Sección 5 del E1: benchmark de los 3 modelos base de instancias y análisis estadístico.

Se puede correr como script (desde la raíz del repo) o pegar como celda del notebook:
    python estadistica_instancias.py [--resultados results/instancias/corridas]

Entradas: <corridas>/<modelo>_s<semilla>/{test_por_imagen.csv, metricas.json}.
Métrica primaria: F1 de instancia por imagen (declaración a priori del EDA). Unidad de análisis: la imagen. Unidad de independencia: el GRUPO (sesión o escena; 14 en test).

1. Por modelo, sobre semillas: media, DE e IC95 (t de Student) de PQ por imagen, PQ por grupo y AP de máscara.
2. Por grupo: F1 medio del grupo (promedio de sus imágenes y luego de las semillas) -> una fila por grupo.
   - Friedman sobre los 3 modelos (bloques = grupos).
   - Wilcoxon pareado por grupo para cada par, con corrección de Holm.
   - IC95 bootstrap de la diferencia media, remuestreando GRUPOS (10 000 réplicas).
   - Efecto mínimo detectable (MDE) aproximado para el par, con alfa de Holm y potencia 0.80.
3. Comparación con δ (umbral práctico para mejora de modelos avanzados).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DELTA = 0.2  # 20 puntos porcentuales de F1, margen práctico del proyecto
METRICA = "f1"  # primaria declarada: F1 de instancia por imagen (IoU >= 0.5, húngaro)
B = 10_000


def cargar(carpeta: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    imgs, runs = [], []
    for d in sorted(carpeta.glob("*_s*")):
        if not (d / "test_por_imagen.csv").exists():
            continue
        modelo, semilla = d.name.rsplit("_s", 1)
        t = pd.read_csv(d / "test_por_imagen.csv")
        t["modelo"], t["semilla"] = modelo, int(semilla)
        imgs.append(t)
        m = json.loads((d / "metricas.json").read_text(encoding="utf8"))["test"]
        runs.append({"modelo": modelo, "semilla": int(semilla), "AP": m["AP"], "AP50": m["AP50"],
                     "F1_img": m.get("F1_img"), "F1_grupo": m.get("F1_grupo"),
                     "PQ_img": m["PQ_img"], "PQ_grupo": m["PQ_grupo"]})
    if not imgs:
        raise FileNotFoundError(f"No hay corridas en {carpeta}")
    return pd.concat(imgs, ignore_index=True), pd.DataFrame(runs)


def ic95(x: pd.Series) -> float:
    n = len(x)
    return float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def holm(p: dict) -> dict:
    orden = sorted(p, key=p.get)
    m, ajust, previo = len(orden), {}, 0.0
    for i, k in enumerate(orden):
        previo = max(previo, min(1.0, (m - i) * p[k]))
        ajust[k] = previo
    return ajust


def bootstrap_grupos(dif: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(dif), size=(B, len(dif)))
    medias = dif[idx].mean(axis=1)
    return float(np.quantile(medias, 0.025)), float(np.quantile(medias, 0.975))


def analizar(carpeta: Path, delta: float = DELTA, metrica: str = METRICA) -> dict:
    imgs, runs = cargar(carpeta)
    salida = {}

    # 1. Resumen por modelo sobre semillas
    res = runs.groupby("modelo").agg(n_semillas=("semilla", "nunique"),
                                     F1_img_media=("F1_img", "mean"), F1_img_de=("F1_img", "std"),
                                     F1_grupo_media=("F1_grupo", "mean"),
                                     PQ_img_media=("PQ_img", "mean"), PQ_img_de=("PQ_img", "std"),
                                     PQ_grupo_media=("PQ_grupo", "mean"), PQ_grupo_de=("PQ_grupo", "std"),
                                     AP_media=("AP", "mean"), AP_de=("AP", "std"), AP50_media=("AP50", "mean"))
    res["F1_img_ic95"] = runs.groupby("modelo").F1_img.apply(ic95)
    res["PQ_img_ic95"] = runs.groupby("modelo").PQ_img.apply(ic95)
    res["AP_ic95"] = runs.groupby("modelo").AP.apply(ic95)
    salida["por_modelo"] = res
    print("== Resumen por modelo (test, sobre semillas)\n", res.round(4).to_string(), "\n")

    # 2. Tabla por grupo: promedio de imágenes del grupo, luego de semillas
    g = (imgs.groupby(["modelo", "semilla", "grupo"])[metrica].mean()
         .groupby(["modelo", "grupo"]).mean().unstack("modelo"))
    salida["por_grupo"] = g
    print(f"== {metrica} por grupo (n = {len(g)} grupos)\n", g.round(3).to_string(), "\n")
    modelos = list(g.columns)
    if len(modelos) >= 3:
        f = stats.friedmanchisquare(*[g[c] for c in modelos])
        salida["friedman"] = {"estadistico": float(f.statistic), "p": float(f.pvalue)}
        print(f"Friedman ({len(modelos)} modelos, {len(g)} grupos): chi2 = {f.statistic:.3f}, p = {f.pvalue:.4f}")

    rng = np.random.default_rng(42)
    pares = list(itertools.combinations(modelos, 2))
    p_crudos, filas = {}, []
    for a, b in pares:
        dif = (g[a] - g[b]).to_numpy()
        w = stats.wilcoxon(g[a], g[b], zero_method="wilcox", method="auto")
        p_crudos[(a, b)] = float(w.pvalue)
        lo, hi = bootstrap_grupos(dif, rng)
        filas.append({"par": f"{a} - {b}", "dif_media": dif.mean(), "de_dif": dif.std(ddof=1),
                      "ic95_boot_grupos": (round(lo, 4), round(hi, 4)), "wilcoxon_W": float(w.statistic),
                      "p": float(w.pvalue), "gana_a_en_grupos": int((dif > 0).sum())})
    ajust = holm(p_crudos)
    n = len(g)
    alfa_holm = 0.05 / len(pares)
    for fila, (a, b) in zip(filas, pares):
        fila["p_holm"] = ajust[(a, b)]
        # MDE aproximado (t pareado; Wilcoxon tiene eficiencia ~0.955 frente a t con datos normales)
        fila["MDE"] = (stats.t.ppf(1 - alfa_holm / 2, n - 1) + stats.t.ppf(0.80, n - 1)) * fila["de_dif"] / np.sqrt(n)
        fila["supera_delta"] = bool(abs(fila["dif_media"]) >= delta and fila["p_holm"] < 0.05)
    t = pd.DataFrame(filas)
    salida["pares"] = t
    print("\n== Pares (unidad = grupo; Holm sobre", len(pares), "contrastes)\n", t.round(4).to_string(index=False))
    print(f"\nδ = {delta}. MDE con alfa de Holm {alfa_holm:.4f} y potencia 0.80. Si MDE > δ, el test no tiene "
          "potencia para detectar una diferencia práctica: se reporta como no concluyente, no como empate.")
    return salida


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default="results/instancias/corridas")
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--metrica", default=METRICA, choices=["f1", "pq", "sq", "dice_semantico"])
    a = ap.parse_args()
    analizar(Path(a.resultados), a.delta, a.metrica)
