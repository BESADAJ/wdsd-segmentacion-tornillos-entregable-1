"""Baseline no profundo (entorno wdsd_dl): regresión logística por píxel + componentes conexas.

Rasgos por píxel (15): RGB y HSV en [0, 1]; sobre el gris, filtro gaussiano, magnitud del gradiente (Sobel del
suavizado) y laplaciano del suavizado, con σ = 1, 2 y 4 px. Se calculan con convoluciones de torch (GPU si hay).

Entrenamiento: todos los píxeles de tornillo de train y 1000 píxeles de fondo por imagen. Los de fondo salen de un
grupo fijo de 4000 por imagen (semilla 0, en caché); cada semilla de la corrida elige sus 1000 de ese grupo, así que
la semilla cambia la muestra de fondo. Rasgos estandarizados con la media y la desviación de la muestra. Pérdida:
entropía cruzada con pesos que igualan las dos clases + λ·||w||², con L-BFGS. La búsqueda (T = 3) es sobre λ.

Instancias: probabilidad > 0.5, componentes conexas con conectividad 8 (en el EDA ningún tornillo toca a otro), área
mínima 5 px; el score de cada instancia es la probabilidad media de sus píxeles. El umbral de score se elige en val
como en todos los modelos.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .comun import DATOS, guardar_preds, imagenes, ruta_json

SIGMAS = (1.0, 2.0, 4.0)
N_POOL, N_FONDO = 4000, 1000
AREA_MIN = 5
CACHE = DATOS / "e2" / "clasico_pool.npz"


def _gauss(sigma: float, disp):
    import torch
    r = int(4 * sigma + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=disp)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def rasgos(ruta: Path, disp):
    """(15, H, W) float32 en el dispositivo."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    im = Image.open(ruta).convert("RGB")
    rgb = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).to(disp)
    hsv = torch.from_numpy(np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0).permute(2, 0, 1).to(disp)
    gris = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])[None, None]
    sob = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=disp)
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=disp)
    canales = [rgb, hsv]
    for s in SIGMAS:
        k = _gauss(s, disp)
        r = (len(k) - 1) // 2
        g = F.conv2d(F.pad(gris, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        g = F.conv2d(F.pad(g, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
        gp = F.pad(g, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(gp, sob.view(1, 1, 3, 3)); gy = F.conv2d(gp, sob.t().reshape(1, 1, 3, 3))
        canales += [g[0], torch.sqrt(gx ** 2 + gy ** 2)[0], F.conv2d(gp, lap.view(1, 1, 3, 3))[0]]
    return torch.cat(canales, 0)


def _mascara_gt(img: dict, anns: list[dict]) -> np.ndarray:
    from pycocotools import mask as mask_utils
    h, w = img["height"], img["width"]
    m = np.zeros((h, w), np.uint8)
    for a in anns:
        m |= mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], h, w)))
    return m


def grupo_de_pixeles(disp, log=print):
    """Rasgos de todos los píxeles de tornillo y de 4000 de fondo por imagen de train (semilla 0). Se guarda en caché."""
    if CACHE.exists():
        d = np.load(CACHE)
        return d["X"], d["y"], d["img"]
    d = json.loads(ruta_json("train").read_text(encoding="utf8"))
    por_img = {}
    for a in d["annotations"]:
        por_img.setdefault(a["image_id"], []).append(a)
    rng = np.random.default_rng(0)
    X, y, idx = [], [], []
    t0 = time.time()
    for k, im in enumerate(d["images"]):
        f = rasgos(Path(im["ruta"]), disp).flatten(1).T.cpu().numpy()  # (H*W, 15)
        gt = _mascara_gt(im, por_img.get(im["id"], [])).ravel().astype(bool)
        pos = np.flatnonzero(gt)
        neg = rng.choice(np.flatnonzero(~gt), N_POOL, replace=False)
        for sel, lab in ((pos, 1), (neg, 0)):
            X.append(f[sel]); y.append(np.full(len(sel), lab, np.uint8)); idx.append(np.full(len(sel), k, np.int32))
        if k % 100 == 0:
            log(f"  rasgos de train: {k}/{len(d['images'])} ({time.time() - t0:.0f} s)")
    X, y, idx = np.concatenate(X), np.concatenate(y), np.concatenate(idx)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X.astype(np.float32), y=y, img=idx)
    return X, y, idx


def ajustar(X: np.ndarray, y: np.ndarray, lam: float, disp):
    import torch
    Xt = torch.from_numpy(X).to(disp)
    yt = torch.from_numpy(y.astype(np.float32)).to(disp)
    mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    Xt = (Xt - mu) / sd
    peso = torch.where(yt > 0, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))  # clases con el mismo peso total
    w = torch.zeros(X.shape[1], device=disp, requires_grad=True)
    b = torch.zeros(1, device=disp, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-7)

    def cierre():
        opt.zero_grad()
        z = Xt @ w + b
        perdida = (peso * torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")).mean() \
            + lam * (w ** 2).sum()
        perdida.backward()
        return perdida

    opt.step(cierre)
    perdida = float(cierre())  # opt.step devuelve la pérdida inicial; esta es la final
    return {"w": w.detach(), "b": b.detach(), "mu": mu, "sd": sd, "perdida": perdida}


def predecir(modelo: dict, split: str, disp) -> list[dict]:
    import torch
    from scipy import ndimage
    from wdsd_inst.metricas import a_rle
    from .comun import carpeta_imagenes
    estructura = np.ones((3, 3), bool)  # conectividad 8
    salida = []
    for im in imagenes(split):
        f = rasgos(carpeta_imagenes(split) / f"{Path(im['file_name']).stem}.png", disp)
        z = torch.einsum("chw,c->hw", (f - modelo["mu"][:, None, None]) / modelo["sd"][:, None, None], modelo["w"]) + modelo["b"]
        p = torch.sigmoid(z).cpu().numpy()
        etiquetas, n = ndimage.label(p > 0.5, structure=estructura)
        if n == 0:
            continue
        areas = np.bincount(etiquetas.ravel(), minlength=n + 1)
        medias = ndimage.mean(p, etiquetas, index=np.arange(1, n + 1))
        candidatos = [(float(medias[i - 1]), i) for i in range(1, n + 1) if areas[i] >= AREA_MIN]
        for s, i in sorted(candidatos, reverse=True)[:100]:
            salida.append({"image_id": im["id"], "category_id": 1, "segmentation": a_rle(etiquetas == i), "score": s})
    return salida


def entrenar(modelo: str, carpeta: Path, semilla: int, epocas: int, factor: float, splits: tuple, disp,
             preentrenado: bool = True, conservar_pesos: bool = True, log=print) -> dict:
    """factor = λ. epocas no se usa (el ajuste es convexo, hasta convergencia de L-BFGS)."""
    import torch
    t0 = time.time()
    X, y, img = grupo_de_pixeles(disp, log)
    rng = np.random.default_rng(semilla)
    sel = [np.flatnonzero(y == 1)]
    for k in np.unique(img):
        neg = np.flatnonzero((img == k) & (y == 0))
        sel.append(rng.choice(neg, min(N_FONDO, len(neg)), replace=False))
    sel = np.concatenate(sel)
    m = ajustar(X[sel], y[sel], factor, disp)
    t_entrenar = time.time() - t0
    log(f"  clasico λ={factor:g}: {int(y[sel].sum())} píxeles de tornillo, {int((y[sel] == 0).sum())} de fondo, "
        f"pérdida {m['perdida']:.4f}")
    if conservar_pesos:
        torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in m.items()}, carpeta / "mejor.pt")
    t1 = time.time()
    for split in splits:
        guardar_preds(carpeta, split, predecir(m, split, disp))
    return {"lambda": factor, "lr_inicial": None, "optimizador": "L-BFGS", "lote": None, "acumulacion": None,
            "t_entrenar_s": t_entrenar, "t_predecir_s": time.time() - t1, "sin_epocas": True, "mejor_epoca": None,
            "parametros": int(m["w"].numel() + 1), "historial": [], "pixeles_entrenamiento": int(len(sel))}
