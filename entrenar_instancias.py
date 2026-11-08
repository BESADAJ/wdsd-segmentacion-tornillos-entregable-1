"""Modelos base de segmentación de instancias en WDSD: Mask R-CNN y YOLOv8-seg (SOLOv2 va aparte).

Uso, desde la raíz del repo (la carpeta que tiene results/eda/particion.csv), con el entorno activado:

  python entrenar_instancias.py preparar --data-root .
  python entrenar_instancias.py piloto   --horas 10
  python entrenar_instancias.py cola     --hasta 07:30 --epocas-maskrcnn 12 --epocas-yolo 40
  python entrenar_instancias.py cola     --hasta 07:30 --semillas 45,46      (otra máquina, otras semillas)
  python entrenar_instancias.py resumen

- preparar: crea datos_inst/coco/{train,val,test}.json con la partición agrupada 744/93/108 y
  datos_inst/yolo/ (enlaces duros a las imágenes + etiquetas en formato YOLO).
- piloto: 1 época de cada modelo (semilla 42); mide segundos por época y memoria pico, y propone épocas
  para que 2 modelos x 5 semillas quepan en --horas.
- cola: corre por rondas (semilla 42 de ambos modelos, luego 43, ...). Salta las corridas terminadas y
  no empieza una que no alcance a terminar antes de --hasta (estimado con el piloto).
- resumen: junta results/instancias/corridas/*/metricas.json en results/instancias/resumen.csv.

Protocolo común: pesos preentrenados en COCO, imagen completa a 1280x720 (sin reescalar ni recortar),
volteo horizontal; mejor época por val; test una sola vez; PQ con el umbral de score elegido en val.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

SEMILLAS = [42, 43, 44, 45, 46]
MODELOS = ["maskrcnn", "yolov8seg"]
DATOS = RAIZ / "datos_inst"
SALIDA = RAIZ / "results" / "instancias"


def fijar_semilla(s: int):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def dispositivo():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entorno() -> dict:
    import torch
    d = {"python": platform.python_version(), "torch": torch.__version__, "so": platform.platform(),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    try:
        import torchvision, ultralytics
        d.update(torchvision=torchvision.__version__, ultralytics=ultralytics.__version__)
    except ImportError:
        pass
    return d


def exigir_gpu(args):
    import torch
    if not torch.cuda.is_available() and not args.permitir_cpu:
        sys.exit(f"ERROR: torch {torch.__version__} no ve la GPU (torch.cuda.is_available() = False). "
                 "Entrenar en CPU tardaría horas. Reinstala torch con CUDA o usa --permitir-cpu solo para pruebas.")


def correr(modelo: str, semilla: int, epocas: int, carpeta: Path, args) -> dict:
    from wdsd_inst import maskrcnn, yolo
    from wdsd_inst.metricas import evaluar_corrida
    exigir_gpu(args)
    fijar_semilla(semilla)
    disp = dispositivo()
    t0 = time.time()
    if modelo == "maskrcnn":
        r = maskrcnn.entrenar(DATOS, carpeta, semilla, epocas, disp, lote=args.lote_maskrcnn,
                              preentrenado=not args.sin_preentrenar, max_iter=args.max_iter)
    elif modelo == "yolov8seg":
        r = yolo.entrenar(DATOS, carpeta, semilla, epocas, disp, variante=args.variante_yolo, lote=args.lote_yolo,
                          preentrenado=not args.sin_preentrenar, fraccion=args.fraccion_yolo)
    else:
        raise ValueError(modelo)
    m = evaluar_corrida(carpeta, DATOS / "coco/val.json", DATOS / "coco/test.json")
    resumen = {"modelo": modelo, "semilla": semilla, "epocas": epocas, "s_total": time.time() - t0,
               "entorno": entorno(), **{k: v for k, v in r.items() if k != "historial"}, "metricas": m}
    if "historial" in r:
        import pandas as pd
        pd.DataFrame(r["historial"]).to_csv(carpeta / "historial.csv", index=False)
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=1, default=str), encoding="utf8")
    print(f"== {modelo} s{semilla}: test F1 img {m['test']['F1_img']:.3f} | AP {m['test']['AP']:.3f} | PQ img {m['test']['PQ_img']:.3f} | "
          f"PQ grupo {m['test']['PQ_grupo']:.3f} | {resumen['s_total'] / 60:.1f} min")
    return resumen


def cmd_preparar(args):
    from wdsd_inst.datos import preparar
    r = preparar(Path(args.data_root).resolve(), RAIZ / "results/eda/particion.csv", DATOS)
    print(json.dumps(r, indent=1))


def cmd_piloto(args):
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas = []
    for m in args.modelos.split(","):
        r = correr(m, 42, 1, SALIDA / "piloto" / m, args)
        s_ep = r.get("s_por_epoca") or r["s_total"]
        filas.append({"modelo": m, "s_corrida_1_epoca": round(r["s_total"]), "gpu_pico_gb": round(r["gpu_pico_gb"], 2),
                      "parametros": r["parametros"], "s_por_epoca_aprox": round(s_ep)})
    import pandas as pd
    t = pd.DataFrame(filas)
    t.to_csv(SALIDA / "piloto" / "piloto.csv", index=False)
    print(t.to_string(index=False))
    seg_por_ronda = t.s_corrida_1_epoca.sum()  # 1 época de cada modelo + evaluación
    rondas = len(SEMILLAS)
    print(f"\nPresupuesto: {args.horas} h para {rondas} semillas x {len(t)} modelos.")
    for e in (5, 10, 15, 20, 30, 40, 50):
        horas = rondas * sum(r.s_por_epoca_aprox * e + (r.s_corrida_1_epoca - r.s_por_epoca_aprox) for r in t.itertuples()) / 3600
        print(f"  {e:3d} épocas por modelo -> {horas:5.1f} h en total ({'cabe' if horas <= args.horas else 'no cabe'})")


def hora_limite(hasta: str) -> dt.datetime:
    hh, mm = map(int, hasta.split(":"))
    ahora = dt.datetime.now()
    lim = ahora.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return lim if lim > ahora else lim + dt.timedelta(days=1)


def cmd_cola(args):
    import pandas as pd
    lim = hora_limite(args.hasta)
    piloto = SALIDA / "piloto" / "piloto.csv"
    est = pd.read_csv(piloto).set_index("modelo") if piloto.exists() else None
    epocas = {"maskrcnn": args.epocas_maskrcnn, "yolov8seg": args.epocas_yolo}
    semillas = [int(s) for s in args.semillas.split(",")]
    for s in semillas:
        for m in args.modelos.split(","):
            carpeta = SALIDA / "corridas" / f"{m}_s{s}"
            if (carpeta / "resumen.json").exists():
                print(f"Salto {m} s{s}: ya terminó."); continue
            if est is not None and m in est.index:
                seg = est.loc[m, "s_por_epoca_aprox"] * epocas[m] * 1.1 + 120
                if dt.datetime.now() + dt.timedelta(seconds=float(seg)) > lim:
                    print(f"Me detengo: {m} s{s} (~{seg / 60:.0f} min) no termina antes de {lim:%H:%M}."); return
            print(f"\n== {dt.datetime.now():%H:%M} empiezo {m} s{s} ({epocas[m]} épocas)")
            correr(m, s, epocas[m], carpeta, args)
    print("Cola terminada.")


def cmd_resumen(args):
    import pandas as pd
    filas = []
    for p in sorted((SALIDA / "corridas").glob("*/resumen.json")):
        r = json.loads(p.read_text(encoding="utf8"))
        filas.append({"modelo": r["modelo"], "semilla": r["semilla"], "epocas": r["epocas"],
                      "min": round(r["s_total"] / 60, 1), "gpu": r["entorno"].get("gpu"),
                      **{f"test_{k}": v for k, v in r["metricas"]["test"].items()},
                      "umbral_score": r["metricas"]["umbral_score"]})
    t = pd.DataFrame(filas)
    t.to_csv(SALIDA / "resumen.csv", index=False)
    print(t.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preparar"); p.add_argument("--data-root", default=".")
    p.set_defaults(f=cmd_preparar)
    for nombre, f in (("piloto", cmd_piloto), ("cola", cmd_cola)):
        p = sub.add_parser(nombre)
        p.add_argument("--modelos", default=",".join(MODELOS))
        p.add_argument("--lote-maskrcnn", type=int, default=2)
        p.add_argument("--lote-yolo", type=int, default=4)
        p.add_argument("--variante-yolo", default="yolov8n-seg")
        p.add_argument("--fraccion-yolo", type=float, default=1.0)
        p.add_argument("--max-iter", type=int, default=None, help="solo para pruebas: iteraciones por época")
        p.add_argument("--sin-preentrenar", action="store_true", help="solo para pruebas sin internet")
        p.add_argument("--permitir-cpu", action="store_true", help="solo para pruebas: deja entrenar sin GPU")
        p.set_defaults(f=f)
        if nombre == "piloto":
            p.add_argument("--horas", type=float, default=10)
        else:
            p.add_argument("--hasta", required=True, help="HH:MM, hora local")
            p.add_argument("--semillas", default=",".join(map(str, SEMILLAS)))
            p.add_argument("--epocas-maskrcnn", type=int, default=12)
            p.add_argument("--epocas-yolo", type=int, default=40)
    p = sub.add_parser("resumen"); p.set_defaults(f=cmd_resumen)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
