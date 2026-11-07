"""Entrenamiento de los tres modelos base (FCN, SegNet, U-Net) en WDSD.

Uso (desde la raíz del repositorio, con el entorno activado):

    python entrenar_base.py piloto  --horas 6.5
    python entrenar_base.py cola    --hasta 07:15 --max-epocas 40

- piloto: 1 época de cada modelo (semilla 42), mide segundos por época y propone
  cuántas épocas caben para 3 modelos x 5 semillas en las horas indicadas.
- cola: corre por rondas (semilla 1 de los tres modelos, luego semilla 2, ...).
  Antes de cada corrida estima si termina antes de --hasta; si no, se detiene.
  Si se interrumpe, al relanzar salta las corridas ya terminadas.

Protocolo (igual para los tres modelos):
- Partición agrupada de results/eda/particion.csv (columna split): 744/93/108.
- Normalización con media y desviación de train (results/eda/normalizacion_train.json).
- Entrenamiento con recortes de 512x512 a resolución nativa; una fracción de los
  recortes se centra cerca de un tornillo (los tornillos son ~0.09 % de los píxeles).
- Pérdida: 0.5 * entropía cruzada binaria + 0.5 * Dice suave.
- Adam, lr 1e-3, coseno; precisión mixta en GPU.
- Validación y prueba sobre la imagen completa 720x1280.
- Selección del mejor epoch por Dice medio por imagen en validación; parada temprana.
- Prueba (test) una sola vez por corrida, con el mejor checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .modelos import MODELOS, crear, n_parametros, profundidad

PROJECT = Path(__file__).resolve().parents[2]
RESULTS_EDA = PROJECT / "results" / "eda"
SALIDA = PROJECT / "results" / "modelos_base"
SEM = "Annotations/Semantic Segmentation"
SEMILLAS = [42, 43, 44, 45, 46]
ORDEN_MODELOS = ["fcn", "segnet", "unet"]


# ----------------------------------------------------------------------------- datos

def raices_datos(data_root: Path | None) -> list[Path]:
    base = data_root or PROJECT.parent
    if (base / "train").is_dir() and (base / SEM).is_dir():
        return [base]
    roots = sorted(base.glob("WDSD-*/WDSD")) + sorted(base.glob("WDSD"))
    if not roots:
        raise FileNotFoundError(f"No encuentro carpetas WDSD-*/WDSD ni WDSD en {base}. Usa --data-root.")
    return roots


def tabla_rutas(data_root: Path | None) -> pd.DataFrame:
    part = pd.read_csv(RESULTS_EDA / "particion.csv")
    img, mask = {}, {}
    for root in raices_datos(data_root):
        for s in ("train", "val", "test"):
            for p in (root / s).glob("*.png"):
                img[(s, p.name)] = str(p)
            for p in (root / SEM / s).glob("*.png"):
                mask[(s, p.name)] = str(p)
    claves = list(zip(part.split_oficial, part.archivo))
    part["ruta_img"] = [img.get(k) for k in claves]
    part["ruta_mask"] = [mask.get(k) for k in claves]
    faltan = part[part.ruta_img.isna() | part.ruta_mask.isna()]
    if len(faltan):
        raise FileNotFoundError(f"Faltan {len(faltan)} imágenes o máscaras, p. ej. {faltan.archivo.head(3).tolist()}")
    return part


def leer_mascara(ruta: str) -> np.ndarray:
    m = np.asarray(Image.open(ruta))
    if m.ndim == 3:
        m = m.max(axis=2)
    return m > 0


def cargar_split(df: pd.DataFrame, split: str):
    sub = df[df.split == split].reset_index(drop=True)
    t0 = time.time()
    imgs = np.stack([np.asarray(Image.open(r).convert("RGB")) for r in sub.ruta_img])  # N,H,W,3 uint8
    masks = np.stack([leer_mascara(r) for r in sub.ruta_mask])  # N,H,W bool
    print(f"  {split}: {len(sub)} imágenes cargadas en {time.time() - t0:.0f} s")
    return sub, imgs, masks


class Recortes:
    """Genera lotes de recortes aleatorios desde las imágenes en memoria (sin DataLoader:
    en Windows los workers copiarían ~3 GB de imágenes a cada proceso)."""

    def __init__(self, imgs, masks, tam=512, frac_tornillo=0.5, rng=None):
        self.imgs, self.masks, self.tam, self.frac = imgs, masks, tam, frac_tornillo
        self.rng = rng or np.random.default_rng(0)
        self.coords = []
        for m in masks:
            ys, xs = np.nonzero(m)
            if len(ys) > 4000:
                sel = np.random.default_rng(0).choice(len(ys), 4000, replace=False)
                ys, xs = ys[sel], xs[sel]
            self.coords.append((ys, xs))

    def recorte(self, i):
        H, W = self.masks[i].shape
        t = self.tam
        ys, xs = self.coords[i]
        if len(ys) and self.rng.random() < self.frac:
            k = self.rng.integers(len(ys))
            cy = ys[k] + self.rng.integers(-t // 3, t // 3 + 1)
            cx = xs[k] + self.rng.integers(-t // 3, t // 3 + 1)
            y0 = int(np.clip(cy - t // 2, 0, H - t))
            x0 = int(np.clip(cx - t // 2, 0, W - t))
        else:
            y0 = int(self.rng.integers(0, H - t + 1))
            x0 = int(self.rng.integers(0, W - t + 1))
        im = self.imgs[i, y0:y0 + t, x0:x0 + t]
        mk = self.masks[i, y0:y0 + t, x0:x0 + t]
        if self.rng.random() < 0.5:
            im, mk = im[:, ::-1], mk[:, ::-1]
        if self.rng.random() < 0.5:
            im, mk = im[::-1], mk[::-1]
        return im, mk

    def epoca(self, recortes_por_imagen, lote):
        orden = np.repeat(np.arange(len(self.imgs)), recortes_por_imagen)
        self.rng.shuffle(orden)
        for a in range(0, len(orden), lote):
            par = [self.recorte(i) for i in orden[a:a + lote]]
            yield np.stack([p[0] for p in par]), np.stack([p[1] for p in par])


# ----------------------------------------------------------------------------- utilidades

def fijar_semilla(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def a_tensor(im_u8: np.ndarray, media, desv, disp) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(im_u8)).to(disp, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255)
    return (x - media) / desv


def perdida(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + y.sum() + 1)
    return 0.5 * bce + 0.5 * dice


def metricas_imagen(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    if tp + fp + fn == 0:  # imagen sin tornillos y predicción vacía
        dice = iou = 1.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
    return {"dice": dice, "iou": iou, "tp": tp, "fp": fp, "fn": fn,
            "precision_px": tp / (tp + fp) if tp + fp else np.nan,
            "recall_px": tp / (tp + fn) if tp + fn else np.nan}


@torch.no_grad()
def evaluar(modelo, imgs, masks, media, desv, disp, amp, lote=2):
    modelo.eval()
    filas = []
    for a in range(0, len(imgs), lote):
        x = a_tensor(imgs[a:a + lote], media, desv, disp)
        with torch.autocast(disp.type, dtype=torch.float16, enabled=amp):
            out = modelo(x)
        pred = (out.float() > 0).squeeze(1).cpu().numpy()
        for j in range(len(pred)):
            filas.append(metricas_imagen(pred[j], masks[a + j]))
    return filas


def entorno(disp) -> dict:
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "so": platform.platform(), "dispositivo": str(disp),
            "gpu": torch.cuda.get_device_name(0) if disp.type == "cuda" else None}


# ----------------------------------------------------------------------------- una corrida

def correr(modelo_nombre, semilla, datos, args, disp, carpeta: Path, max_epocas=None, limite_s=None):
    fijar_semilla(semilla)
    (sub_tr, im_tr, mk_tr), (sub_va, im_va, mk_va), (sub_te, im_te, mk_te) = datos
    norm = json.loads((RESULTS_EDA / "normalizacion_train.json").read_text())
    media = torch.tensor(norm["media_rgb_0_1"], device=disp).view(1, 3, 1, 1)
    desv = torch.tensor(norm["std_rgb_0_1"], device=disp).view(1, 3, 1, 1)
    amp = disp.type == "cuda"

    modelo = crear(modelo_nombre, args.ancho).to(disp)
    max_epocas = max_epocas or args.max_epocas
    opt = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epocas)
    scaler = torch.amp.GradScaler(disp.type, enabled=amp)
    gen = Recortes(im_tr, mk_tr, args.recorte, args.frac_tornillo, np.random.default_rng(semilla))
    if disp.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    carpeta.mkdir(parents=True, exist_ok=True)
    hist, mejor, sin_mejora, cortada = [], -1.0, 0, False
    t_inicio = time.time()
    t_entreno = 0.0
    for ep in range(1, max_epocas + 1):
        modelo.train()
        t0 = time.time(); suma = 0.0; n = 0
        for xb, yb in gen.epoca(args.recortes_por_imagen, args.lote):
            x = a_tensor(xb, media, desv, disp)
            y = torch.from_numpy(np.ascontiguousarray(yb)).to(disp).unsqueeze(1).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(disp.type, dtype=torch.float16, enabled=amp):
                out = modelo(x)
            l = perdida(out.float(), y)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            suma += l.item() * len(xb); n += len(xb)
        sched.step()
        if disp.type == "cuda":
            torch.cuda.synchronize()
        t_ep = time.time() - t0
        t_entreno += t_ep
        t1 = time.time()
        val = evaluar(modelo, im_va, mk_va, media, desv, disp, amp)
        t_val = time.time() - t1
        dice_val = float(np.mean([f["dice"] for f in val]))
        hist.append({"epoca": ep, "perdida_train": suma / n, "dice_val": dice_val,
                     "t_epoca_s": round(t_ep, 2), "t_val_s": round(t_val, 2), "lr": sched.get_last_lr()[0]})
        print(f"    {modelo_nombre} s{semilla} ep{ep:3d}  pérdida {suma / n:.4f}  Dice val {dice_val:.4f}  "
              f"({t_ep:.0f}+{t_val:.0f} s)", flush=True)
        if dice_val > mejor + 1e-4:
            mejor, sin_mejora = dice_val, 0
            torch.save(modelo.state_dict(), carpeta / "mejor.pt")
        else:
            sin_mejora += 1
        if sin_mejora >= args.paciencia:
            break
        if limite_s and time.time() - t_inicio > limite_s:
            print("    tope de tiempo de la corrida alcanzado")
            cortada = True
            break

    pd.DataFrame(hist).to_csv(carpeta / "historial.csv", index=False)
    modelo.load_state_dict(torch.load(carpeta / "mejor.pt", map_location=disp, weights_only=True))
    t2 = time.time()
    test = evaluar(modelo, im_te, mk_te, media, desv, disp, amp)
    if disp.type == "cuda":
        torch.cuda.synchronize()
    t_test = time.time() - t2
    por_img = pd.DataFrame(test)
    por_img.insert(0, "archivo", sub_te.archivo.values)
    por_img.insert(1, "grupo", sub_te.grupo.values)
    por_img.insert(0, "semilla", semilla)
    por_img.insert(0, "modelo", modelo_nombre)
    por_img.to_csv(carpeta / "test_por_imagen.csv", index=False)

    resumen = {
        "modelo": modelo_nombre, "semilla": semilla, "parametros": n_parametros(modelo),
        "capas_conv": profundidad(modelo), "ancho_base": args.ancho,
        "epocas": len(hist), "cortada_por_tiempo": cortada, "mejor_epoca": int(np.argmax([h["dice_val"] for h in hist]) + 1),
        "dice_val_mejor": mejor, "dice_test_medio": float(por_img.dice.mean()),
        "iou_test_medio": float(por_img.iou.mean()),
        "t_entreno_s": round(t_entreno, 1), "t_total_s": round(time.time() - t_inicio, 1),
        "s_por_epoca": round(t_entreno / len(hist), 2),
        "t_test_s": round(t_test, 2), "ms_por_imagen_test": round(1000 * t_test / len(por_img), 1),
        "mem_pico_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if disp.type == "cuda" else None,
        "config": {k: v for k, v in vars(args).items() if k != "func"},
        "entorno": entorno(disp), "fin": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=1, default=str), encoding="utf8")
    return resumen


# ----------------------------------------------------------------------------- comandos

def preparar(args):
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {disp}" + (f" ({torch.cuda.get_device_name(0)})" if disp.type == "cuda" else
                                     "  <-- SIN GPU: el entrenamiento será muy lento"))
    df = tabla_rutas(Path(args.data_root) if args.data_root else None)
    print("Cargando imágenes en memoria...")
    datos = tuple(cargar_split(df, s) for s in ("train", "val", "test"))
    return disp, datos


def cmd_piloto(args):
    disp, datos = preparar(args)
    filas = []
    for m in ORDEN_MODELOS:
        r = correr(m, 42, datos, args, disp, SALIDA / "piloto" / m, max_epocas=1)
        t_ep = r["s_por_epoca"] + float(pd.read_csv(SALIDA / "piloto" / m / "historial.csv").t_val_s.iloc[0])
        filas.append({"modelo": m, "parametros": r["parametros"], "capas_conv": r["capas_conv"],
                      "s_por_epoca_con_val": round(t_ep, 1), "mem_pico_gpu_gb": r["mem_pico_gpu_gb"]})
    tabla = pd.DataFrame(filas)
    seg_epoca_3 = tabla.s_por_epoca_con_val.sum()  # una época de cada modelo
    presupuesto = args.horas * 3600 * 0.9  # 10 % de margen
    epocas = int(presupuesto // (seg_epoca_3 * len(SEMILLAS)))
    print("\n" + tabla.to_string(index=False))
    print(f"\nCon {args.horas} h y {len(SEMILLAS)} semillas caben ~{epocas} épocas por corrida (con 10 % de margen).")
    print(f"Sugerencia: python entrenar_base.py cola --hasta HH:MM --max-epocas {max(1, min(epocas, args.max_epocas))}")
    tabla.to_csv(SALIDA / "piloto" / "piloto.csv", index=False)


def hora_limite(hasta: str) -> dt.datetime:
    hh, mm = map(int, hasta.split(":"))
    ahora = dt.datetime.now()
    limite = ahora.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return limite + dt.timedelta(days=1) if limite <= ahora else limite


def cmd_curva(args):
    """Curva de aprendizaje: U-Net (semilla 42) con fracciones anidadas de los GRUPOS de train.

    Responde si quitar datos de entrenamiento (p. ej., para pasar grupos a test) costaría Dice.
    Se mide en validación; la prueba no se usa para decidir. Orden 1.0, 0.5, 0.75 para tener
    los extremos primero. El tiempo restante se reparte por igual entre las fracciones pendientes.
    """
    limite = hora_limite(args.hasta)
    disp, datos = preparar(args)
    (sub_tr, im_tr, mk_tr), val, test = datos
    grupos = np.array(sorted(sub_tr.grupo.unique()))
    orden = np.random.default_rng(42).permutation(grupos)  # subconjuntos anidados: 50 % ⊂ 75 % ⊂ 100 %
    filas = []
    fracs = [float(f) for f in args.fracciones.split(",")]
    for i, f in enumerate(fracs):
        carpeta = SALIDA / "curva" / f"{args.modelo}_f{int(round(f * 100))}"
        if (carpeta / "resumen.json").exists():
            r = json.loads((carpeta / "resumen.json").read_text(encoding="utf8"))
            print(f"  ya existe {carpeta.name}, se salta")
        else:
            elegidos = set(orden[: max(1, int(round(f * len(grupos))))])
            m = sub_tr.grupo.isin(elegidos).to_numpy()
            sub = (sub_tr[m].reset_index(drop=True), im_tr[m], mk_tr[m])
            restante = (limite - dt.datetime.now()).total_seconds()
            cuota = restante / (len(fracs) - i)
            if cuota < 300:
                print(f"  quedan {restante / 60:.0f} min: no alcanza para {carpeta.name}. Fin.")
                break
            print(f"\n== curva {args.modelo} {f:.0%} de train: {m.sum()} imágenes, {len(elegidos)} grupos "
                  f"(cuota {cuota / 60:.0f} min)")
            r = correr(args.modelo, 42, (sub, val, test), args, disp, carpeta, limite_s=cuota - 60)
            r["n_img_train"], r["n_grupos_train"] = int(m.sum()), len(elegidos)
            (carpeta / "resumen.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf8")
        filas.append({"fraccion": f, "n_img_train": r.get("n_img_train"), "n_grupos_train": r.get("n_grupos_train"),
                      "dice_val_mejor": round(r["dice_val_mejor"], 4), "epocas": r["epocas"],
                      "mejor_epoca": r["mejor_epoca"], "cortada_por_tiempo": r.get("cortada_por_tiempo")})
    tabla = pd.DataFrame(filas).sort_values("fraccion")
    print("\n" + tabla.to_string(index=False))
    tabla.to_csv(SALIDA / "curva" / "curva.csv", index=False)
    print("Ojo: si alguna fila dice cortada_por_tiempo=True, esa fracción no terminó de entrenar.")


def cmd_cola(args):
    limite = hora_limite(args.hasta)
    ahora = dt.datetime.now()
    print(f"Cola hasta {limite:%Y-%m-%d %H:%M} ({(limite - ahora).total_seconds() / 3600:.1f} h)")
    disp, datos = preparar(args)
    s_ep = {}
    piloto = SALIDA / "piloto" / "piloto.csv"
    if piloto.exists():
        s_ep = dict(pd.read_csv(piloto)[["modelo", "s_por_epoca_con_val"]].values)
    registro = SALIDA / "corridas.csv"
    for semilla in SEMILLAS[: args.semillas]:
        for m in ORDEN_MODELOS:
            carpeta = SALIDA / f"{m}_s{semilla}"
            if (carpeta / "resumen.json").exists():
                print(f"  ya existe {carpeta.name}, se salta")
                continue
            restante = (limite - dt.datetime.now()).total_seconds()
            estimado = s_ep.get(m, 0) * args.max_epocas
            if estimado and estimado > restante:
                print(f"  {m} s{semilla}: estimado {estimado / 60:.0f} min > restante {restante / 60:.0f} min. Fin de la cola.")
                return
            print(f"\n== {m}  semilla {semilla}  (restan {restante / 60:.0f} min)")
            r = correr(m, semilla, datos, args, disp, carpeta, limite_s=max(60, restante - 120))
            s_ep[m] = r["t_total_s"] / r["epocas"]
            fila = {k: v for k, v in r.items() if k not in ("config", "entorno")}
            nuevo = not registro.exists()
            with open(registro, "a", newline="", encoding="utf8") as f:
                w = csv.DictWriter(f, fieldnames=list(fila))
                if nuevo:
                    w.writeheader()
                w.writerow(fila)
    print("Cola terminada.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    for nombre, func in (("piloto", cmd_piloto), ("cola", cmd_cola), ("curva", cmd_curva)):
        p = sub.add_parser(nombre)
        p.set_defaults(func=func)
        p.add_argument("--data-root", default=None, help="carpeta con WDSD-*/WDSD (por defecto, la carpeta padre del repo)")
        p.add_argument("--ancho", type=int, default=16, help="canales de la primera etapa (U-Net reducida = 16)")
        p.add_argument("--recorte", type=int, default=512)
        p.add_argument("--frac-tornillo", type=float, default=0.5)
        p.add_argument("--recortes-por-imagen", type=int, default=2)
        p.add_argument("--lote", type=int, default=8)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--max-epocas", type=int, default=40)
        p.add_argument("--paciencia", type=int, default=8)
    sub.choices["piloto"].add_argument("--horas", type=float, default=6.5)
    sub.choices["cola"].add_argument("--hasta", default="07:15", help="hora local límite HH:MM")
    sub.choices["cola"].add_argument("--semillas", type=int, default=5)
    sub.choices["curva"].add_argument("--hasta", default="07:20", help="hora local límite HH:MM")
    sub.choices["curva"].add_argument("--fracciones", default="1.0,0.5,0.75")
    sub.choices["curva"].add_argument("--modelo", default="unet", choices=list(MODELOS))
    args = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
