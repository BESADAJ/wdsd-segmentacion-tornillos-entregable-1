"""Inventario, trazabilidad y medidas por imagen de WDSD.

El dataset llega partido en dos descargas de Google Drive (``WDSD-*-1-001`` y
``WDSD-*-1-002``). Este módulo las une en un solo inventario y calcula, con
caché en ``results/eda``, lo que el notebook de EDA necesita.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.fft import dctn
from scipy.sparse.csgraph import connected_components
from skimage import measure

PROJECT = Path(__file__).resolve().parents[2]
ENTREGABLE = PROJECT.parent
RESULTS = PROJECT / "results" / "eda"
SEED = 42
H, W = 720, 1280

SEM = "Annotations/Semantic Segmentation"
VIS = "Annotations/Visualization"
COCO = "Annotations/Object Detection & Instance Segmentation COCO-format"
COCO_FILES = {"train": "Training_set.json", "val": "Validation_set.json", "test": "Test_set.json"}


def data_roots() -> list[Path]:
    """Carpetas WDSD de cada parte descargada, en orden."""
    roots = sorted(ENTREGABLE.glob("WDSD-*/WDSD"))
    if not roots:
        raise FileNotFoundError(f"No hay carpetas WDSD-*/WDSD en {ENTREGABLE}")
    return roots


def nombre_grupo(stem: str) -> str:
    """Grupo que se deduce del nombre: sesión de grabación o familia sin sesión."""
    m = re.fullmatch(r"(?i)(micro|tower|screen)_(\d+)_cam3_png_color_\d+", stem)
    if m:
        return f"{m[1].lower()}_{m[2]}"
    m = re.fullmatch(r"pc_(\d+)_\d+", stem)
    if m:
        return f"pc_{m[1]}"
    m = re.fullmatch(r"(diaplay|eco)(\d+)_\d+", stem)
    if m:
        return f"{m[1]}{m[2]}"
    m = re.fullmatch(r"(frame|screws)_\d+", stem)
    if m:
        return m[1]
    if stem.isdigit():
        return "numerico"
    raise ValueError(f"Nombre no reconocido: {stem}")


def familia(grupo: str) -> str:
    return re.sub(r"[_\d]+.*$", "", grupo) or grupo


def numero_frame(stem: str) -> int:
    return int(re.search(r"(\d+)$", stem)[1])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventario(refresh: bool = False) -> pd.DataFrame:
    """Una fila por archivo de imagen con sus anotaciones y hashes."""
    cache = RESULTS / "inventario.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    filas = []
    for parte, root in enumerate(data_roots(), start=1):
        for split in ("train", "val", "test"):
            for p in sorted((root / split).glob("*.png")):
                filas.append({"split_oficial": split, "archivo": p.name, "parte_img": parte, "ruta_img": str(p)})
    df = pd.DataFrame(filas)
    masks, vis = {}, {}
    for parte, root in enumerate(data_roots(), start=1):
        for split in ("train", "val", "test"):
            for p in (root / SEM / split).glob("*.png"):
                masks[(split, p.name)] = (parte, str(p))
            for p in (root / VIS / split).glob("*.png"):
                vis[(split, p.name.split("__", 1)[1])] = str(p)
    claves = list(zip(df.split_oficial, df.archivo))
    df["parte_mask"] = [masks.get(k, (np.nan, None))[0] for k in claves]
    df["ruta_mask"] = [masks.get(k, (np.nan, None))[1] for k in claves]
    df["ruta_vis"] = [vis.get(k) for k in claves]
    df["stem"] = df.archivo.str[:-4]
    df["grupo_nombre"] = df.stem.map(nombre_grupo)
    df["familia"] = df.grupo_nombre.map(familia)
    df["n_frame"] = df.stem.map(numero_frame)
    df["sha256_img"] = df.ruta_img.map(lambda p: _sha256(Path(p)))
    df["sha256_mask"] = df.ruta_mask.map(lambda p: _sha256(Path(p)))
    df.to_csv(cache, index=False)
    return df


def huella_dataset(df: pd.DataFrame) -> str:
    """SHA-256 del listado ordenado de hashes de imagen y máscara."""
    filas = sorted(f"{a}|{b}|{c}" for a, b, c in zip(df.archivo, df.sha256_img, df.sha256_mask))
    return hashlib.sha256("\n".join(filas).encode()).hexdigest()


def phash(path: str) -> np.ndarray:
    """Hash perceptual DCT de 255 bits: 16x16 coeficientes menos el de continua."""
    im = np.asarray(Image.open(path).convert("L").resize((64, 64), Image.BILINEAR), float)
    d = dctn(im, norm="ortho")[:16, :16].ravel()[1:]
    return d > np.median(d)


def matriz_phash(df: pd.DataFrame, refresh: bool = False) -> np.ndarray:
    """Distancias de Hamming entre todas las imágenes (orden de ``df``)."""
    cache = RESULTS / "phash.npy"
    if cache.exists() and not refresh:
        hashes = np.load(cache)
    else:
        hashes = np.array([phash(p) for p in df.ruta_img])
        np.save(cache, hashes)
    h = hashes.astype(np.int16)
    return (h[:, None, :] != h[None, :, :]).sum(2)


def medidas_imagen(df: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Formato, intensidad, nitidez y fondo de cada imagen."""
    cache = RESULTS / "medidas_imagen.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    filas = []
    for fila in df.itertuples():
        with Image.open(fila.ruta_img) as img:
            modo, tam = img.mode, img.size
            a = np.asarray(img.convert("RGB"), float)
        with Image.open(fila.ruta_mask) as mk:
            modo_m, tam_m = mk.mode, mk.size
        g = a.mean(2)
        r_, g_, b_ = a[..., 0], a[..., 1], a[..., 2]
        filas.append({
            "archivo": fila.archivo, "ancho": tam[0], "alto": tam[1], "modo": modo,
            "ancho_m": tam_m[0], "alto_m": tam_m[1], "modo_m": modo_m,
            "r_media": r_.mean(), "g_media": g_.mean(), "b_media": b_.mean(),
            "r_std": r_.std(), "g_std": g_.std(), "b_std": b_.std(),
            "gris_media": g.mean(), "contraste": g.std(),
            "sat_alta": (g >= 250).mean(), "sat_baja": (g <= 5).mean(),
            "nitidez": ndi.laplace(g).var(),
            "frac_verde": ((g_ > 1.15 * r_) & (g_ > 1.15 * b_)).mean(),
        })
    out = pd.DataFrame(filas)
    out.to_csv(cache, index=False)
    return out


def histogramas_rgb(rutas) -> np.ndarray:
    """Histograma acumulado (3, 256) sobre las rutas dadas."""
    h = np.zeros((3, 256), np.int64)
    for p in rutas:
        a = np.asarray(Image.open(p).convert("RGB"))
        for c in range(3):
            h[c] += np.bincount(a[..., c].ravel(), minlength=256)
    return h


def medidas_mascara(df: pd.DataFrame, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla por imagen y por componente conexa de las máscaras PNG."""
    c_img, c_ins = RESULTS / "medidas_mascara.csv", RESULTS / "instancias.csv"
    if c_img.exists() and c_ins.exists() and not refresh:
        return pd.read_csv(c_img), pd.read_csv(c_ins)
    img_f, ins_f = [], []
    anillo = ndi.generate_binary_structure(2, 2)
    for fila in df.itertuples():
        raw = np.asarray(Image.open(fila.ruta_mask))
        m = raw > 0
        gray = np.asarray(Image.open(fila.ruta_img).convert("L"), float)
        lab = measure.label(m, connectivity=2)
        props = measure.regionprops(lab)
        img_f.append({
            "archivo": fila.archivo, "ndim": raw.ndim, "valores": ",".join(map(str, np.unique(raw))),
            "frac_fg": m.mean(), "px_fg": int(m.sum()), "n_comp": len(props),
        })
        for r in props:
            y0, x0, y1, x1 = r.bbox
            p = 6
            caja = (max(0, y0 - p), max(0, x0 - p), min(H, y1 + p), min(W, x1 + p))
            sub = lab[caja[0]:caja[2], caja[1]:caja[3]] == r.label
            dentro = gray[caja[0]:caja[2], caja[1]:caja[3]][sub]
            aro = ndi.binary_dilation(sub, anillo, iterations=4) & ~ndi.binary_dilation(sub, anillo)
            fuera = gray[caja[0]:caja[2], caja[1]:caja[3]][aro]
            ins_f.append({
                "archivo": fila.archivo, "area": int(r.area), "cy": r.centroid[0], "cx": r.centroid[1],
                "alto": y1 - y0, "ancho": x1 - x0, "solidez": r.solidity,
                "excentricidad": r.eccentricity, "circularidad": 4 * np.pi * r.area / max(r.perimeter, 1) ** 2,
                "toca_borde": y0 == 0 or x0 == 0 or y1 == H or x1 == W,
                "media_dentro": dentro.mean(), "media_aro": fuera.mean() if fuera.size else np.nan,
                "std_aro": fuera.std() if fuera.size else np.nan,
            })
    a, b = pd.DataFrame(img_f), pd.DataFrame(ins_f)
    a.to_csv(c_img, index=False)
    b.to_csv(c_ins, index=False)
    return a, b


def carga_coco() -> dict[str, dict]:
    salida = {}
    for root in data_roots():
        for split, nombre in COCO_FILES.items():
            p = root / COCO / nombre
            if p.exists():
                salida[split] = json.loads(p.read_text(encoding="utf8"))
    return salida


def rasteriza_coco(anns: list[dict]) -> np.ndarray:
    lienzo = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(lienzo)
    for a in anns:
        for poli in a["segmentation"]:
            d.polygon(poli, fill=1, outline=1)
    return np.asarray(lienzo) > 0


def acuerdo_coco_png(df: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Compara, imagen por imagen, las dos versiones de la anotación."""
    cache = RESULTS / "acuerdo_coco_png.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    coco = carga_coco()
    rutas = {(f.split_oficial, f.archivo): f.ruta_mask for f in df.itertuples()}
    filas = []
    for split, d in coco.items():
        por_img = defaultdict(list)
        for a in d["annotations"]:
            por_img[a["image_id"]].append(a)
        for im in d["images"]:
            anns = por_img[im["id"]]
            ruta = rutas.get((split, im["file_name"]))
            if ruta is None:
                filas.append({"split_oficial": split, "archivo": im["file_name"], "sin_par": True})
                continue
            m = np.asarray(Image.open(ruta)) > 0
            c = rasteriza_coco(anns)
            inter, union = (m & c).sum(), (m | c).sum()
            filas.append({
                "split_oficial": split, "archivo": im["file_name"], "sin_par": False,
                "n_coco": len(anns), "n_comp_png": measure.label(m, connectivity=2).max(),
                "iou": inter / max(union, 1), "dice": 2 * inter / max(m.sum() + c.sum(), 1),
                "px_png": int(m.sum()), "px_coco": int(c.sum()), "area_coco": sum(a["area"] for a in anns),
            })
    out = pd.DataFrame(filas)
    out.to_csv(cache, index=False)
    return out


BLOQUE_FRAME = 200


def grupos_independencia(df: pd.DataFrame, dist: np.ndarray, umbral: int) -> np.ndarray:
    """Componentes conexas del grafo de dependencia entre imágenes.

    Une dos imágenes si comparten sesión en el nombre, si son de ``frame`` y caen
    en el mismo bloque de numeración, o si su distancia pHash es <= ``umbral``.
    """
    clave = df.grupo_nombre.astype(str).copy()
    es_frame = df.grupo_nombre.eq("frame")
    clave[es_frame] = "frame_b" + (df.n_frame[es_frame] // BLOQUE_FRAME).astype(str)
    es_screws = df.grupo_nombre.eq("screws")
    clave[es_screws] = "screws_" + df.stem[es_screws]
    codigos = pd.factorize(clave)[0]
    adyacencia = (codigos[:, None] == codigos[None, :]) | (dist <= umbral)
    return connected_components(adyacencia, directed=False)[1]


def particion_agrupada(df: pd.DataFrame, grupo: np.ndarray, fracs=(0.8, 0.1, 0.1),
                       reinicios: int = 300, semilla: int = SEED) -> np.ndarray:
    """Asigna grupos completos a train/val/test imitando las fracciones por familia.

    Búsqueda local con reinicios sobre el error cuadrático por familia y total.
    Una familia con tres o más grupos debe aportar a cada partición al menos el
    40 % de lo que le corresponde. Solo usa familia y tamaño de grupo: ninguna
    información de la etiqueta.
    """
    rng = np.random.default_rng(semilla)
    fams = sorted(df.familia.unique())
    conteo = pd.crosstab(grupo, df.familia)[fams].to_numpy(float)
    total_fam, total = conteo.sum(0), conteo.sum()
    objetivo = np.array(fracs)
    exigible = (conteo > 0).sum(0) >= 3
    minimo = 0.4 * objetivo[:, None] * total_fam

    def costo(suma: np.ndarray) -> float:
        por_familia = ((suma / total_fam - objetivo[:, None]) ** 2 * total_fam / total).sum()
        deficit = np.clip(minimo - suma, 0, None)[:, exigible].sum() / total
        return por_familia + 5 * ((suma.sum(1) / total - objetivo) ** 2).sum() + 10 * deficit

    mejor, mejor_costo = None, np.inf
    for _ in range(reinicios):
        asignacion = rng.choice(3, size=len(conteo), p=objetivo)
        suma = np.array([conteo[asignacion == k].sum(0) for k in range(3)])
        actual = costo(suma)
        hubo_mejora = True
        while hubo_mejora:
            hubo_mejora = False
            for g in rng.permutation(len(conteo)):
                origen = asignacion[g]
                for destino in {0, 1, 2} - {origen}:
                    tentativa = suma.copy()
                    tentativa[origen] -= conteo[g]
                    tentativa[destino] += conteo[g]
                    c = costo(tentativa)
                    if c < actual - 1e-12:
                        suma, actual, origen = tentativa, c, destino
                        asignacion[g] = destino
                        hubo_mejora = True
        if actual < mejor_costo:
            mejor, mejor_costo = asignacion.copy(), actual
    return np.array(["train", "val", "test"], dtype=object)[mejor[grupo]]


def huella_particion(archivos, sha_img) -> str:
    filas = sorted(f"{a}|{s}" for a, s in zip(archivos, sha_img))
    return hashlib.sha256("\n".join(filas).encode()).hexdigest()


ESCALAS = {"1280x720": (1280, 720), "960x540": (960, 540), "640x360": (640, 360),
           "512x288": (512, 288), "512x512": (512, 512), "320x180": (320, 180), "256x256": (256, 256)}


def efecto_resolucion(rutas_mascara) -> pd.DataFrame:
    """Instancias que sobreviven al reducir y techo de Dice por resolución."""
    filas = []
    for ruta in rutas_mascara:
        m = np.asarray(Image.open(ruta)) > 0
        lab = measure.label(m, connectivity=2)
        props = measure.regionprops(lab)
        base = Image.fromarray((m * 255).astype(np.uint8))
        for nombre, (w, h) in ESCALAS.items():
            bajo = np.asarray(base.resize((w, h), Image.BOX)) >= 128
            arriba = np.asarray(Image.fromarray((bajo * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
            vivos = 0
            for r in props:
                yy = (r.coords[:, 0] * h // H).astype(int)
                xx = (r.coords[:, 1] * w // W).astype(int)
                vivos += bool(bajo[yy, xx].any())
            filas.append({"escala": nombre, "archivo": Path(ruta).name, "n": len(props), "vivos": vivos,
                          "dice_techo": 2 * (m & arriba).sum() / max(m.sum() + arriba.sum(), 1)})
    return pd.DataFrame(filas)


def analisis_parches(rutas_mascara, tamanos=(256, 384, 512)) -> pd.DataFrame:
    """Parches sin solape: vacíos, densidad de primer plano e instancias cortadas."""
    filas = []
    for t in tamanos:
        vacios = total = fg_px = px_nv = ins = cortes = 0
        for ruta in rutas_mascara:
            m = np.asarray(Image.open(ruta)) > 0
            ys = list(range(0, H - t + 1, t)) or [0]
            xs = list(range(0, W - t + 1, t))
            if ys[-1] + t < H:
                ys.append(H - t)
            if xs[-1] + t < W:
                xs.append(W - t)
            for y in ys:
                for x in xs:
                    c = m[y:y + t, x:x + t].sum()
                    total += 1
                    vacios += c == 0
                    if c:
                        fg_px += c
                        px_nv += t * t
            for r in measure.regionprops(measure.label(m, connectivity=2)):
                y0, x0, y1, x1 = r.bbox
                ins += 1
                cortes += (y0 // t != (y1 - 1) // t) or (x0 // t != (x1 - 1) // t)
        filas.append({"parche": t, "parches": total, "pct_vacios": 100 * vacios / total,
                      "pct_fg_no_vacios": 100 * fg_px / max(px_nv, 1), "pct_inst_cortadas": 100 * cortes / ins})
    return pd.DataFrame(filas)


def perfil_alineacion(filas, desplazamientos=range(-8, 9)) -> pd.DataFrame:
    """Gradiente medio sobre el borde de la máscara al desplazarla en x e y."""
    salida = []
    for ruta_img, ruta_mask in filas:
        g = np.asarray(Image.open(ruta_img).convert("L"), float)
        grad = np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1))
        m = np.asarray(Image.open(ruta_mask)) > 0
        borde = m & ~ndi.binary_erosion(m)
        ys, xs = np.nonzero(borde)
        for d in desplazamientos:
            salida.append({"d": d, "eje": "x", "grad": grad[ys, np.clip(xs + d, 0, W - 1)].mean()})
            salida.append({"d": d, "eje": "y", "grad": grad[np.clip(ys + d, 0, H - 1), xs].mean()})
    return pd.DataFrame(salida).groupby(["eje", "d"]).grad.mean().reset_index()


def sensibilidad_borde(rutas_mascara) -> pd.DataFrame:
    """Dice por imagen entre la máscara y perturbaciones de un píxel de sí misma."""
    filas = []
    for ruta in rutas_mascara:
        m = np.asarray(Image.open(ruta)) > 0
        variantes = {
            "desplazada 1 px (x)": np.roll(m, 1, axis=1),
            "desplazada 1 px (x e y)": np.roll(m, (1, 1), axis=(0, 1)),
            "erosionada 1 px": ndi.binary_erosion(m),
            "dilatada 1 px": ndi.binary_dilation(m),
        }
        fila = {"archivo": Path(ruta).name}
        for k, v in variantes.items():
            fila[k] = 2 * (m & v).sum() / max(m.sum() + v.sum(), 1)
        filas.append(fila)
    return pd.DataFrame(filas)
