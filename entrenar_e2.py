"""Entrenamiento del E2: piloto de tiempo, búsqueda de hiperparámetros y corridas finales (preregistro_E2.md §3–§4).

Cada modelo corre en el entorno de su marco: este script es el orquestador (solo biblioteca estándar, sirve cualquier
python) y lanza cada corrida como un proceso aparte con el python del entorno que toca. Antes de nada, una vez por
máquina (desde la raíz del repo, después de `entrenar_instancias.py preparar`):

  python entrenar_e2.py configurar --nombre 3060 --py-dl <python de wdsd_dl> --py-mmdet <python de wdsd_mmdet>
         --py-yolo9 <python de wdsd_yolo9> --repo-yolov7 <...\\yolov7u7\\seg> --repo-yolov9 <...\\yolov9>
  python entrenar_e2.py datos                     # submuestra fija de 93 imágenes de train (semilla 0)

Después, en este orden:

  python entrenar_e2.py piloto  --modelos todos                   # 1 época, semilla 42: solo tiempos, sin métricas
  python entrenar_e2.py buscar  --modelos todos --epocas-final 12 # T = 3 configuraciones, semilla 42, E/3 épocas
  python entrenar_e2.py final   --modelos todos --epocas 12 --semillas 42,43,44 --hasta "2026-11-19 20:00"
  python entrenar_e2.py resumen

- Las corridas terminadas (con resumen.json) se saltan, así que se puede interrumpir y volver a lanzar.
- Una corrida que falla se repite una vez con la misma semilla; si vuelve a fallar queda fallo.json y se sigue.
- final corre por rondas: todos los modelos con la primera semilla, después la segunda... Con --hasta no empieza una
  corrida que el piloto de esta máquina estima que no termina antes de esa hora.
- Salidas: results/e2/{piloto/<máquina>, busqueda, corridas}/<modelo>_.../ con preds_*.json, historial.csv,
  metricas.json, resumen.json y log.txt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
from wdsd_e2 import registro as R  # noqa: E402

SALIDA = RAIZ / "results" / "e2"
MAQUINA = RAIZ / "maquina_e2.json"
DATOS = RAIZ / "datos_inst"
N_TRAIN93 = 93
SPLITS = {"piloto": ("val",), "buscar": ("val",), "final": ("val", "test", "train93")}


# ---------------------------------------------------------------- configuración de la máquina

def cmd_configurar(a):
    pys = {"dl": a.py_dl, "mmdet": a.py_mmdet, "yolo9": a.py_yolo9}
    pruebas = {
        "dl": "import torch, torchvision, ultralytics, scipy, pandas, pycocotools; "
              "print(torch.__version__, ultralytics.__version__, torch.cuda.is_available(), "
              "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')",
        "mmdet": "import torch, mmdet, mmcv, mmengine, scipy, pandas, pycocotools; "
                 "print(torch.__version__, mmdet.__version__, torch.cuda.is_available())",
        "yolo9": "import torch, cv2, yaml, scipy, pandas, pycocotools; print(torch.__version__, cv2.__version__, torch.cuda.is_available())",
    }
    ok = True
    for k, py in pys.items():
        r = subprocess.run([py, "-c", pruebas[k]], capture_output=True, text=True, encoding="utf8", errors="replace")
        print(f"{k:6s} {py}\n       {(r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else '(sin salida)'}")
        ok &= r.returncode == 0 and ("True" in r.stdout or a.permitir_cpu)
    repos = {"yolov7": a.repo_yolov7, "yolov9": a.repo_yolov9}
    for k, ruta in repos.items():
        spec = R.MODELOS[k]
        t = Path(ruta) / "segment" / "train.py"
        parche = t.exists() and any(l.strip().startswith("nbs = 2") for l in t.read_text(encoding="utf8").splitlines())
        pesos = (Path(ruta) / spec["pesos"]).exists()
        print(f"{k:6s} {ruta}\n       segment/train.py {'con' if parche else 'SIN'} nbs = 2; {spec['pesos']} {'existe' if pesos else 'NO EXISTE'}")
        ok &= parche and pesos
    MAQUINA.write_text(json.dumps({"nombre": a.nombre, "python": pys, "repos": repos}, indent=1), encoding="utf8")
    print(f"\n{'Todo bien' if ok else 'HAY PROBLEMAS (arriba)'}. Configuración en {MAQUINA}")
    if not ok:
        sys.exit(1)


def maquina() -> dict:
    if not MAQUINA.exists():
        sys.exit(f"Falta {MAQUINA.name}: corre primero `entrenar_e2.py configurar` en esta máquina.")
    return json.loads(MAQUINA.read_text(encoding="utf8"))


def cmd_datos(a):
    tr = json.loads((DATOS / "coco" / "train.json").read_text(encoding="utf8"))
    ids = sorted(random.Random(0).sample(sorted(i["id"] for i in tr["images"]), N_TRAIN93))
    sub = {"images": [i for i in tr["images"] if i["id"] in ids],
           "annotations": [x for x in tr["annotations"] if x["image_id"] in ids], "categories": tr["categories"]}
    (DATOS / "coco" / "train93.json").write_text(json.dumps(sub), encoding="utf8")
    lista = [(DATOS / "yolo" / "images" / "train" / f"{Path(i['file_name']).stem}.png").resolve().as_posix()
             for i in sub["images"]]
    (DATOS / "yolo" / "train93.txt").write_text("\n".join(lista) + "\n", encoding="utf8")
    print(f"train93: {len(sub['images'])} imágenes, {len(sub['annotations'])} tornillos, "
          f"{len({i['grupo'] for i in sub['images']})} grupos -> datos_inst/coco/train93.json y datos_inst/yolo/train93.txt")


# ---------------------------------------------------------------- lanzar una corrida

def lanzar(modelo: str, modo: str, semilla: int, epocas: int, valor: float, carpeta: Path, a, cfg: dict) -> bool:
    """Corre una corrida en su entorno. Devuelve True si terminó bien (o ya estaba terminada)."""
    if (carpeta / "resumen.json").exists():
        print(f"Salto {carpeta.name}: ya terminó.")
        return True
    spec = R.MODELOS[modelo]
    cmd = [cfg["python"][spec["entorno"]], str(RAIZ / "entrenar_e2.py"), "_una", "--modelo", modelo, "--modo", modo,
           "--semilla", str(semilla), "--epocas", str(epocas), "--valor", repr(valor), "--carpeta", str(carpeta)]
    if spec["marco"] == "oficial":
        cmd += ["--repo", cfg["repos"][spec["repo"]]]
    cmd += ["--permitir-cpu"] * a.permitir_cpu + ["--sin-preentrenar"] * a.sin_preentrenar
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "WDSD_MAQUINA": cfg["nombre"]}
    for intento in (1, 2):
        if carpeta.exists():
            shutil.rmtree(carpeta)
        carpeta.mkdir(parents=True)
        print(f"\n== {dt.datetime.now():%Y-%m-%d %H:%M} {carpeta.relative_to(SALIDA)} (intento {intento})", flush=True)
        t0 = time.time()
        with (carpeta / "log.txt").open("wb") as log, subprocess.Popen(cmd, cwd=RAIZ, env=env, stdout=subprocess.PIPE,
                                                                       stderr=subprocess.STDOUT) as p:
            while True:
                trozo = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
                if not trozo:
                    break
                log.write(trozo); log.flush()
                sys.stdout.write(trozo.decode("utf8", errors="replace")); sys.stdout.flush()
            p.wait()
        if p.returncode == 0 and (carpeta / "resumen.json").exists():
            r = json.loads((carpeta / "resumen.json").read_text(encoding="utf8"))
            r["s_reloj"] = time.time() - t0  # incluye arrancar python e importar el marco
            (carpeta / "resumen.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf8")
            return True
        print(f"!! {carpeta.name} falló (código {p.returncode}).")
    fallo = {"modelo": modelo, "modo": modo, "semilla": semilla, "maquina": cfg["nombre"],
             "fecha": dt.datetime.now().isoformat(timespec="seconds"), "log": "log.txt"}
    (carpeta / "fallo.json").write_text(json.dumps(fallo, indent=1), encoding="utf8")
    return False


# ---------------------------------------------------------------- estimación de tiempo con el piloto

def leer_piloto(cfg: dict) -> dict:
    est = {}
    for p in (SALIDA / "piloto" / cfg["nombre"]).glob("*/resumen.json"):
        r = json.loads(p.read_text(encoding="utf8"))
        s_ep = 0.0 if r.get("sin_epocas") else r["t_entrenar_s"]
        fijo = r.get("s_reloj", r["s_total"]) - r["t_predecir_s"] - s_ep
        est[r["modelo"]] = {"s_epoca": s_ep, "s_fijo": fijo, "s_pred_img": r["t_predecir_s"] / r["n_pred_imagenes"]}
    return est


def estimar(e: dict, epocas: int, n_img: int) -> float:
    return e["s_fijo"] + e["s_epoca"] * epocas + e["s_pred_img"] * n_img


def limite(hasta: str | None):
    if not hasta:
        return None
    ahora = dt.datetime.now()
    if len(hasta) <= 5:
        hh, mm = map(int, hasta.split(":"))
        lim = ahora.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return lim if lim > ahora else lim + dt.timedelta(days=1)
    return dt.datetime.fromisoformat(hasta)


def cabe(modelo: str, epocas: int, n_img: int, est: dict, lim) -> bool:
    if lim is None or modelo not in est:
        return True
    s = estimar(est[modelo], epocas, n_img) * 1.1 + 60
    if dt.datetime.now() + dt.timedelta(seconds=s) > lim:
        print(f"Me detengo: {modelo} (~{s / 60:.0f} min) no termina antes de {lim:%Y-%m-%d %H:%M}.")
        return False
    return True


# ---------------------------------------------------------------- comandos

def n_imagenes(split: str) -> int:
    return len(json.loads((DATOS / "coco" / f"{split}.json").read_text(encoding="utf8"))["images"])


def cmd_piloto(a):
    cfg = maquina()
    carpeta = SALIDA / "piloto" / cfg["nombre"]
    for m in R.resolver(a.modelos):
        if m == "vacio":
            continue
        valores, defecto = R.configs(m)
        lanzar(m, "piloto", R.SEMILLA_BUSQUEDA, 1, defecto, carpeta / m, a, cfg)
    est = leer_piloto(cfg)
    n_val = n_imagenes("val")
    n_fin = n_val + n_imagenes("test") + N_TRAIN93
    filas = ["modelo,s_epoca,s_fijo,s_pred_img,gpu_pico_gb,parametros"]
    print(f"\nPiloto en {cfg['nombre']} (segundos):")
    for m, e in sorted(est.items()):
        r = json.loads((carpeta / m / "resumen.json").read_text(encoding="utf8"))
        pico = r.get("gpu_pico_reservada_gb") or r.get("gpu_mem_log_pico_gb")
        filas.append(f"{m},{e['s_epoca']:.0f},{e['s_fijo']:.0f},{e['s_pred_img']:.3f},{pico},{r.get('parametros')}")
        print(f"  {m:12s} época {e['s_epoca'] / 60:6.1f} min | fijo {e['s_fijo'] / 60:5.1f} min | "
              f"predicción {e['s_pred_img']:.2f} s/img | memoria {pico} GB")
    (carpeta / "piloto.csv").write_text("\n".join(filas) + "\n", encoding="utf8")
    print("\nHoras estimadas en esta máquina (todos los modelos del piloto):")
    print("   E | búsqueda (3 config., E/3 épocas) | una semilla de corridas finales")
    for E in range(12, 3, -1):
        hb = sum(3 * estimar(e, R.epocas_busqueda(E), n_val) for e in est.values()) / 3600
        hs = sum(estimar(e, E, n_fin) for e in est.values()) / 3600
        print(f"  {E:2d} | {hb:6.1f} h | {hs:6.1f} h")


def cmd_buscar(a):
    cfg = maquina()
    lim = limite(a.hasta)
    est = leer_piloto(cfg)
    eb = R.epocas_busqueda(a.epocas_final)
    for m in R.resolver(a.modelos):
        if m == "vacio":
            continue
        valores, defecto = R.configs(m)
        for v in valores:
            carpeta = SALIDA / "busqueda" / f"{m}_c{R.etiqueta(v)}"
            if not (carpeta / "resumen.json").exists() and not cabe(m, eb, n_imagenes("val"), est, lim):
                return
            lanzar(m, "buscar", R.SEMILLA_BUSQUEDA, eb, v, carpeta, a, cfg)
        elegir(m)


def elegir(modelo: str):
    """Mejor configuración por F1 de instancia por grupo en val; empate a 0.005 -> la de por defecto."""
    valores, defecto = R.configs(modelo)
    res = {}
    for v in valores:
        p = SALIDA / "busqueda" / f"{modelo}_c{R.etiqueta(v)}" / "resumen.json"
        if not p.exists():
            print(f"{modelo}: falta la configuración {v:g}; no elijo todavía.")
            return
        r = json.loads(p.read_text(encoding="utf8"))
        res[v] = {"F1_grupo_val": r["metricas"]["val"]["F1_grupo"], "AP_val": r["metricas"]["val"]["AP"],
                  "maquina": r.get("maquina"), "mejor_epoca": r.get("mejor_epoca")}
    mejor = max(res, key=lambda v: res[v]["F1_grupo_val"])
    elegido = defecto if res[defecto]["F1_grupo_val"] >= res[mejor]["F1_grupo_val"] - R.EMPATE else mejor
    salida = {"modelo": modelo, "elegido": elegido, "por_defecto": defecto, "regla": "max F1_grupo val; empate 0.005",
              "configuraciones": {R.etiqueta(v): x for v, x in res.items()},
              "misma_maquina": len({x["maquina"] for x in res.values()}) == 1}
    (SALIDA / "busqueda" / f"{modelo}_elegida.json").write_text(json.dumps(salida, indent=1), encoding="utf8")
    print(f"{modelo}: elegida {elegido:g} " + " | ".join(f"{v:g}: F1 {x['F1_grupo_val']:.3f}" for v, x in res.items()))


def cmd_final(a):
    cfg = maquina()
    lim = limite(a.hasta)
    est = leer_piloto(cfg)
    modelos = R.resolver(a.modelos)
    elegidos = {}
    for m in modelos:
        if m == "vacio":
            continue
        p = SALIDA / "busqueda" / f"{m}_elegida.json"
        if not p.exists():
            sys.exit(f"Falta {p.relative_to(RAIZ)}: corre antes `buscar` para {m} (o copia ese archivo de la otra máquina).")
        elegidos[m] = json.loads(p.read_text(encoding="utf8"))["elegido"]
    n_fin = n_imagenes("val") + n_imagenes("test") + N_TRAIN93
    lanzar("vacio", "final", 0, 0, 0.0, SALIDA / "corridas" / "vacio_s0", a, cfg)  # baseline trivial: una vez, segundos
    for s in [int(x) for x in a.semillas.split(",")]:
        for m in elegidos:
            carpeta = SALIDA / "corridas" / f"{m}_s{s}"
            if not (carpeta / "resumen.json").exists() and not cabe(m, a.epocas, n_fin, est, lim):
                return
            lanzar(m, "final", s, a.epocas, elegidos[m], carpeta, a, cfg)
    print("Cola terminada.")


def cmd_resumen(a):
    import pandas as pd
    for nombre in ("corridas", "busqueda"):
        filas = []
        for p in sorted((SALIDA / nombre).glob("*/resumen.json")):
            r = json.loads(p.read_text(encoding="utf8"))
            m = r.get("metricas") or {}
            fila = {k: r.get(k) for k in ("modelo", "semilla", "valor", "epocas", "mejor_epoca", "maquina", "lote",
                                           "acumulacion", "lr_inicial", "parametros", "s_total")}
            fila["grupo"] = R.MODELOS[r["modelo"]]["grupo"]
            fila["familia"] = R.MODELOS[r["modelo"]]["familia"]
            fila["gpu"] = r.get("entorno", {}).get("gpu")
            fila["umbral_score"] = m.get("umbral_score")
            for parte in ("val", "test", "train93"):
                for k, v in (m.get(parte) or {}).items():
                    fila[f"{parte}_{k}"] = v
            filas.append(fila)
        fallos = [json.loads(p.read_text(encoding="utf8")) for p in sorted((SALIDA / nombre).glob("*/fallo.json"))]
        if filas:
            t = pd.DataFrame(filas)
            t.to_csv(SALIDA / f"{nombre}.csv", index=False)
            cols = [c for c in ("modelo", "semilla", "valor", "mejor_epoca", "val_F1_grupo", "test_F1_grupo", "s_total") if c in t]
            print(f"\n{nombre}: {len(t)} corridas -> results/e2/{nombre}.csv")
            print(t[cols].to_string(index=False))
        if fallos:
            print(f"\n{nombre}: {len(fallos)} corridas fallidas:", ", ".join(f"{f['modelo']} s{f['semilla']}" for f in fallos))


# ---------------------------------------------------------------- trabajador (corre dentro del entorno del marco)

def cmd_una(a):
    import importlib
    from wdsd_e2 import comun, evaluar
    carpeta = Path(a.carpeta)
    spec = R.MODELOS[a.modelo]
    disp = comun.preparar_gpu(a.permitir_cpu)
    comun.fijar_semilla(a.semilla)
    t0 = time.time()
    splits = SPLITS[a.modo]
    if spec["marco"] == "vacio":
        r = {"t_entrenar_s": 0.0, "t_predecir_s": 0.0, "sin_epocas": True, "historial": [], "parametros": 0}
        for s in splits:
            comun.guardar_preds(carpeta, s, [])
    else:
        mod = importlib.import_module(f"wdsd_e2.m_{spec['marco']}")
        extra = {"repo": a.repo} if spec["marco"] == "oficial" else {}
        r = mod.entrenar(a.modelo, carpeta, a.semilla, a.epocas, a.valor, splits, disp,
                         preentrenado=not a.sin_preentrenar, conservar_pesos=a.modo == "final", log=print, **extra)
    hist = r.pop("historial", [])
    perdidas = [h.get("perdida") for h in hist if h.get("perdida") is not None]
    if any(not math.isfinite(x) for x in perdidas):  # fallo técnico (preregistro §6): se repite una vez
        raise RuntimeError(f"pérdida no finita en el entrenamiento: {perdidas}")
    if hist:
        import pandas as pd
        pd.DataFrame(hist).to_csv(carpeta / "historial.csv", index=False)
    metricas = None
    if a.modo == "buscar":
        metricas = evaluar.busqueda(carpeta)
    elif a.modo == "final":
        metricas = evaluar.final(carpeta)
    resumen = {"modelo": a.modelo, "modo": a.modo, "semilla": a.semilla, "epocas": a.epocas, "valor": a.valor,
               "maquina": os.environ.get("WDSD_MAQUINA"), "fecha_fin": dt.datetime.now().isoformat(timespec="seconds"),
               "s_total": time.time() - t0, "n_pred_imagenes": sum(n_imagenes(s) for s in splits),
               "entorno": comun.entorno(), **r, "metricas": metricas}
    if hist and spec["marco"] != "clasico":
        ap = [h.get("val_AP") for h in hist if h.get("val_AP") is not None]
        # subentrenado (preregistro §4): la curva de val sigue subiendo en la última época
        resumen["val_AP_sube_al_final"] = bool(len(ap) >= 2 and ap[-1] > max(ap[:-1]))
    comun.escribir_json(carpeta / "resumen.json", resumen)  # se escribe al final: marca la corrida como terminada
    if metricas:
        v = metricas["val"]
        t = metricas.get("test") or {}
        print(f"== {a.modelo} {a.modo} s{a.semilla} valor {a.valor:g}: val F1 grupo {v['F1_grupo']:.3f}"
              + (f" | test F1 grupo {t['F1_grupo']:.3f}" if t else "") + f" | {resumen['s_total'] / 60:.1f} min")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("configurar")
    for k in ("nombre", "py-dl", "py-mmdet", "py-yolo9", "repo-yolov7", "repo-yolov9"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--permitir-cpu", action="store_true")
    p.set_defaults(f=cmd_configurar)
    sub.add_parser("datos").set_defaults(f=cmd_datos)

    def comunes(p):
        p.add_argument("--modelos", default="todos", help="todos | top10 | base | nombres separados por comas")
        p.add_argument("--permitir-cpu", action="store_true", help="solo pruebas")
        p.add_argument("--sin-preentrenar", action="store_true", help="solo pruebas sin internet")
        p.add_argument("--hasta", default=None, help="'HH:MM' o 'AAAA-MM-DD HH:MM', hora local")

    p = sub.add_parser("piloto"); comunes(p); p.set_defaults(f=cmd_piloto)
    p = sub.add_parser("buscar"); comunes(p); p.add_argument("--epocas-final", type=int, required=True)
    p.set_defaults(f=cmd_buscar)
    p = sub.add_parser("final"); comunes(p); p.add_argument("--epocas", type=int, required=True)
    p.add_argument("--semillas", default=",".join(map(str, R.SEMILLAS)))
    p.set_defaults(f=cmd_final)
    sub.add_parser("resumen").set_defaults(f=cmd_resumen)

    p = sub.add_parser("_una", help=argparse.SUPPRESS)
    p.add_argument("--modelo", required=True); p.add_argument("--modo", required=True, choices=list(SPLITS))
    p.add_argument("--semilla", type=int, required=True); p.add_argument("--epocas", type=int, required=True)
    p.add_argument("--valor", type=float, required=True); p.add_argument("--carpeta", required=True)
    p.add_argument("--repo", default=None)
    p.add_argument("--permitir-cpu", action="store_true"); p.add_argument("--sin-preentrenar", action="store_true")
    p.set_defaults(f=cmd_una)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
