"""Prueba de humo del E2: memoria en 6 GB + prueba de sanidad del PDF E2 §11.1 ("sobreajustar un lote pequeño").

Para cada modelo candidato al Top 10:
  1. entrena con 4 imágenes de train, a 1280x720, con AMP y lote 2 (si se queda sin memoria, reintenta con lote 1);
  2. mide la memoria pico de GPU (asignada y reservada);
  3. evalúa en esas mismas 4 imágenes: si el modelo está bien implementado, debe memorizarlas (AP50 de máscara alto).

Uso (desde la raíz del repo, después de `entrenar_instancias.py preparar`). Corridas del E2, en orden (PowerShell):
  # MMDetection (entorno wdsd_mmdet); 300 iteraciones
  & $PYM prueba_humo_e2.py mmdet --modelos cascade,htc,pointrend,msrcnn,scnet,queryinst,condinst,mask2former
  # ultralytics (entorno wdsd_dl); 150 épocas = 300 iteraciones con lote 2
  & $PY prueba_humo_e2.py ultra --modelos yolo11n-seg,yolo26n-seg
  # Repetición con 3 veces más pasos para los que no llegaron a 0.80 (desviación declarada; umbral sin cambio)
  & $PYM prueba_humo_e2.py mmdet --modelos mask2former --iters 900 --lote 1
  & $PYM prueba_humo_e2.py mmdet --modelos condinst --iters 900
  & $PY prueba_humo_e2.py ultra --modelos yolo11n-seg,yolo26n-seg --epocas 450
  # Control: el ganador del E1 con el mismo protocolo
  & $PY prueba_humo_e2.py ultra --modelos yolov8n-seg,yolo11n-seg,yolo26n-seg
  # Datos para YOLOv7/YOLOv9 (repos oficiales; ver results/e2/prueba_humo/repos_oficiales/)
  & $PY prueba_humo_e2.py datos

Resultados: results/e2/prueba_humo/<modelo>/resumen.json (última corrida) y una línea por intento en
results/e2/prueba_humo/tabla_v2.csv.
Criterio declarado (fijado antes de correr): CABE si entrena sin error de memoria con lote >= 1 en 6 GB;
SANIDAD OK si el AP50 de máscara en las 4 imágenes de entrenamiento es >= 0.80 al final.

v2 (2026-11-15, tras la primera corrida):
- Tope de memoria: torch.cuda.set_per_process_memory_fraction(0.95). Sin tope, Windows deja que la GPU use RAM del
  sistema cuando se acaba la VRAM ("sysmem fallback"), así que nunca hay error de memoria y "CABE" no significa nada
  (Mask2Former reservó 8.21 GB en una tarjeta de 6 GB).
- Pico real: mmengine reinicia el pico de memoria cada vez que escribe el log (cada 25 iteraciones), así que el pico
  medido al final solo cubría el último tramo. Ahora se desactiva ese reinicio durante la corrida.
- Anclas de 8 a 128 px en los modelos con RPN (Cascade, HTC, PointRend, MS R-CNN, SCNet), igual que la Mask R-CNN del E1.
  Las oficiales van de 32 a 512 px y un tornillo de 9x8 px no alcanza IoU 0.3 con ninguna, así que la RPN no aprende
  a proponerlos.
- ultralytics: nbs = lote. Por defecto acumula gradientes hasta un lote nominal de 64 (nbs=64), o sea un paso del
  optimizador cada 32 iteraciones con lote 2: en 300 iteraciones casi no aprendía (AP50 = 0 en YOLO11 y YOLO26).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import time
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos_inst"
SALIDA = RAIZ / "results" / "e2" / "prueba_humo"
N_IMG = 4
FRACCION_VRAM = 0.95  # tope del asignador de PyTorch; el resto queda para el contexto CUDA
ESCALA_ANCLA = 2      # tamaño de ancla = escala x paso del nivel FPN (4..64) -> 8..128 px; oficial 8 -> 32..512 px

# nombre corto -> (config oficial dentro de mmdet/.mim/configs, pesos COCO oficiales)
D = "https://download.openmmlab.com/mmdetection/"
MMDET = {
    "cascade": ("cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py",
                D + "v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"),
    # HTC sin rama semántica: nuestra anotación no tiene COCO-stuff. Pesos del HTC 1x (la rama semántica se ignora al cargar).
    "htc": ("htc/htc-without-semantic_r50_fpn_1x_coco.py",
            D + "v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317-7332cf16.pth"),
    "pointrend": ("point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py",
                  D + "v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth"),
    "msrcnn": ("ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py",
               D + "v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth"),
    "queryinst": ("queryinst/queryinst_r50_fpn_1x_coco.py",
                  D + "v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth"),
    "condinst": ("condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py",
                 D + "v3.0/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance/condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223-4c186406.pth"),
    "mask2former": ("mask2former/mask2former_r50_8xb2-lsj-50e_coco.py",
                    D + "v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth"),
    # reservas
    "scnet": ("scnet/scnet_r50_fpn_1x_coco.py",
              D + "v2.0/scnet/scnet_r50_fpn_1x_coco/scnet_r50_fpn_1x_coco-c3f09857.pth"),
    "solo": ("solo/solo_r50_fpn_1x_coco.py",
             D + "v2.0/solo/solo_r50_fpn_1x_coco/solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth"),
}


def subconjunto() -> Path:
    """Escribe datos_inst/coco/humo4.json con las 4 primeras imágenes de train que tienen >= 3 tornillos."""
    destino = DATOS / "coco" / "humo4.json"
    tr = json.loads((DATOS / "coco" / "train.json").read_text(encoding="utf8"))
    n = {}
    for a in tr["annotations"]:
        n[a["image_id"]] = n.get(a["image_id"], 0) + 1
    ids = [im["id"] for im in sorted(tr["images"], key=lambda i: i["id"]) if n.get(im["id"], 0) >= 3][:N_IMG]
    sub = {"images": [i for i in tr["images"] if i["id"] in ids],
           "annotations": [a for a in tr["annotations"] if a["image_id"] in ids],
           "categories": tr["categories"]}
    destino.write_text(json.dumps(sub), encoding="utf8")
    print(f"Subconjunto: {len(sub['images'])} imágenes, {len(sub['annotations'])} tornillos -> {destino}")
    return destino


def _poner(obj, clave, valor):
    """Cambia recursivamente toda clave `clave` dentro de dicts/listas del config."""
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k == clave:
                obj[k] = valor
            else:
                _poner(obj[k], clave, valor)
    elif isinstance(obj, list):
        for x in obj:
            _poner(x, clave, valor)


def config_mmdet(nombre: str, carpeta: Path, lote: int, iters: int, ann: Path):
    import mmdet
    from mmengine.config import Config
    rel, pesos = MMDET[nombre]
    cfg = Config.fromfile(str(Path(mmdet.__file__).parent / ".mim" / "configs" / rel))
    _poner(cfg.model, "num_classes", 1)
    if nombre == "mask2former":
        _poner(cfg.model, "num_things_classes", 1)
        _poner(cfg.model, "num_stuff_classes", 0)
        cfg.model.panoptic_head.loss_cls.class_weight = [1.0, 0.1]
        cfg.model.data_preprocessor.batch_augments = None  # sin el relleno fijo a 1024x1024
    img_dir = (DATOS / "yolo" / "images" / "train").resolve().as_posix() + "/"
    tren = [dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
            dict(type="Resize", scale=(1280, 720), keep_ratio=True), dict(type="PackDetInputs")]
    prueba = [dict(type="LoadImageFromFile"), dict(type="Resize", scale=(1280, 720), keep_ratio=True),
              dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"))]
    ds = lambda pipe, test: dict(type="CocoDataset", data_root="", metainfo=dict(classes=("tornillo",)),
                                 ann_file=ann.resolve().as_posix(), data_prefix=dict(img=img_dir), test_mode=test,
                                 filter_cfg=dict(filter_empty_gt=not test, min_size=0), pipeline=pipe)
    cfg.train_dataloader = dict(batch_size=lote, num_workers=0, persistent_workers=False,
                                sampler=dict(type="InfiniteSampler", shuffle=True), dataset=ds(tren, False))
    cfg.val_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False, drop_last=False,
                              sampler=dict(type="DefaultSampler", shuffle=False), dataset=ds(prueba, True))
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator = dict(type="CocoMetric", metric="segm", ann_file=ann.resolve().as_posix(), format_only=False)
    cfg.test_evaluator = cfg.val_evaluator
    cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=iters, val_interval=iters)
    cfg.val_cfg = dict(type="ValLoop")
    cfg.test_cfg = dict(type="TestLoop")
    rpn = cfg.model.get("rpn_head")
    if rpn is not None and "anchor_generator" in rpn:
        rpn.anchor_generator.scales = [ESCALA_ANCLA]
    if nombre == "scnet":  # sin rama semántica (necesita COCO-stuff, que no tenemos), igual que HTC
        cfg.model.roi_head.semantic_roi_extractor = None
        cfg.model.roi_head.semantic_head = None
    # Optimizador oficial de cada modelo. Con SGD la tasa se escala linealmente al lote (recetas oficiales: lote 16);
    # con AdamW (QueryInst, Mask2Former) se deja la oficial, como es práctica habitual con Adam.
    cfg.optim_wrapper.type = "AmpOptimWrapper"
    if cfg.optim_wrapper.optimizer.type == "SGD":
        cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr * lote / 16
    cfg.param_scheduler = [dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=20)]
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=10**9, save_last=False)
    cfg.default_hooks.logger = dict(type="LoggerHook", interval=25)
    cfg.load_from = pesos
    cfg.resume = False
    cfg.randomness = dict(seed=42, deterministic=False)
    cfg.env_cfg = dict(cudnn_benchmark=False,
                       dist_cfg=dict(backend="gloo" if platform.system() == "Windows" else "nccl"))
    cfg.work_dir = str(carpeta.resolve())
    cfg.visualizer = dict(type="DetLocalVisualizer", vis_backends=[dict(type="LocalVisBackend")],
                          name=f"vis_{nombre}_{lote}_{time.time_ns()}")
    return cfg


def una_mmdet(nombre: str, lote: int, iters: int, ann: Path) -> dict:
    import torch
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    register_all_modules()
    carpeta = SALIDA / nombre / f"lote{lote}"
    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True)
    import gc
    gc.collect()  # suelta lo que quedó de un intento anterior sin memoria antes de reintentar con lote 1
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    reiniciar = torch.cuda.reset_peak_memory_stats
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None  # mmengine lo llama en cada log; queremos el pico total
    t0 = time.time()
    try:
        runner = Runner.from_cfg(config_mmdet(nombre, carpeta, lote, iters, ann))
        runner.train()
        m = runner.val_loop.run()  # métricas en las 4 imágenes de entrenamiento
    finally:
        torch.cuda.reset_peak_memory_stats = reiniciar
    return {"segm_AP": float(m.get("coco/segm_mAP", float("nan"))), "segm_AP50": float(m.get("coco/segm_mAP_50", float("nan"))),
            "s_total": time.time() - t0,
            "pico_asignada_gb": torch.cuda.max_memory_allocated() / 2**30,
            "pico_reservada_gb": torch.cuda.max_memory_reserved() / 2**30,
            "parametros": sum(p.numel() for p in runner.model.parameters())}


def preparar_yolo_humo(ann: Path) -> Path:
    """Copia las mismas 4 imágenes (y sus etiquetas YOLO) a datos_inst/humo_yolo/ y escribe data.yaml (train = val)."""
    sub = json.loads(ann.read_text(encoding="utf8"))
    raiz = DATOS / "humo_yolo"
    for t in ("images", "labels"):
        (raiz / t / "train").mkdir(parents=True, exist_ok=True)
    for im in sub["images"]:
        st = Path(im["file_name"]).stem
        shutil.copy2(DATOS / "yolo" / "images" / "train" / f"{st}.png", raiz / "images" / "train" / f"{st}.png")
        shutil.copy2(DATOS / "yolo" / "labels" / "train" / f"{st}.txt", raiz / "labels" / "train" / f"{st}.txt")
    y = raiz / "data.yaml"
    y.write_text(f"path: {raiz.resolve().as_posix()}\ntrain: images/train\nval: images/train\nnames:\n  0: tornillo\n",
                 encoding="utf8")
    return y


def una_ultra(variante: str, lote: int, epocas: int, yaml: Path) -> dict:
    import torch
    from ultralytics import YOLO
    carpeta = SALIDA / variante / f"lote{lote}"
    if carpeta.exists():
        shutil.rmtree(carpeta)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    modelo = YOLO(f"{variante}.pt")
    modelo.train(data=str(yaml), imgsz=1280, epochs=epocas, batch=lote, seed=42, workers=0, device=0, amp=True,
                 project=str(carpeta.parent.resolve()), name=carpeta.name, exist_ok=True, plots=False, verbose=False,
                 mosaic=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, scale=0.0, translate=0.0,
                 close_mosaic=0, patience=epocas, val=False,
                 nbs=lote)  # sin esto acumula hasta lote 64: con 4 imágenes serían ~10-20 pasos del optimizador en total
    r = modelo.val(data=str(yaml), imgsz=1280, batch=1, device=0, plots=False, verbose=False)
    print(f"-> {variante}: AP50 de CAJA en las 4 imágenes {float(r.box.map50):.3f} (máscara {float(r.seg.map50):.3f})")
    return {"segm_AP": float(r.seg.map), "segm_AP50": float(r.seg.map50), "s_total": time.time() - t0,
            "pico_asignada_gb": torch.cuda.max_memory_allocated() / 2**30,
            "pico_reservada_gb": torch.cuda.max_memory_reserved() / 2**30,
            "parametros": sum(p.numel() for p in modelo.model.parameters())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("marco", choices=["mmdet", "ultra", "datos"], help="datos = solo prepara humo4.json y datos_inst/humo_yolo/")
    ap.add_argument("--modelos", default="", help="lista separada por comas")
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--iters", type=int, default=300, help="mmdet: iteraciones (300 con lote 2 = 150 pasadas por las 4 imágenes)")
    ap.add_argument("--epocas", type=int, default=150, help="ultralytics: épocas sobre las 4 imágenes")
    a = ap.parse_args()

    if a.marco == "datos":
        print("data.yaml para YOLOv7/YOLOv9:", preparar_yolo_humo(subconjunto()))
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(f"torch {torch.__version__} no ve la GPU. Revisa el entorno antes de seguir.")
    gpu = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(FRACCION_VRAM, 0)
    print(f"GPU: {gpu} ({total:.1f} GB, tope {FRACCION_VRAM * total:.2f} GB) | torch {torch.__version__}")
    SALIDA.mkdir(parents=True, exist_ok=True)
    ann = subconjunto()
    yaml = preparar_yolo_humo(ann) if a.marco == "ultra" else None

    tabla = SALIDA / "tabla_v2.csv"
    nuevo = not tabla.exists()
    with tabla.open("a", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "modelo", "marco", "gpu", "lote", "estado", "pico_asignada_gb", "pico_reservada_gb",
                        "segm_AP50_4img", "segm_AP_4img", "sanidad_ok", "minutos", "parametros", "error"])
        for nombre in [m.strip() for m in a.modelos.split(",") if m.strip()]:
            for lote in sorted({a.lote, 1}, reverse=True):
                print(f"\n===== {nombre} | lote {lote} =====")
                try:
                    r = una_mmdet(nombre, lote, a.iters, ann) if a.marco == "mmdet" else una_ultra(nombre, lote, a.epocas, yaml)
                    ok = r["segm_AP50"] >= 0.80
                    fila = [time.strftime("%Y-%m-%d %H:%M"), nombre, a.marco, gpu, lote, "CABE",
                            round(r["pico_asignada_gb"], 2), round(r["pico_reservada_gb"], 2), round(r["segm_AP50"], 3),
                            round(r["segm_AP"], 3), "sí" if ok else "NO", round(r["s_total"] / 60, 1), r["parametros"], ""]
                    print(f"-> {nombre}: CABE con lote {lote} | pico {r['pico_asignada_gb']:.2f} GB "
                          f"(reservada {r['pico_reservada_gb']:.2f}) | AP50 máscara 4 img {r['segm_AP50']:.3f} | "
                          f"sanidad {'OK' if ok else 'NO'}")
                    w.writerow(fila); f.flush()
                    (SALIDA / nombre / "resumen.json").write_text(json.dumps(dict(zip(
                        ["fecha", "modelo", "marco", "gpu", "lote", "estado", "pico_asignada_gb", "pico_reservada_gb",
                         "segm_AP50_4img", "segm_AP_4img", "sanidad_ok", "minutos", "parametros", "error"], fila)),
                        indent=1, default=str), encoding="utf8")
                    break
                except Exception as e:
                    oom = "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError)
                    estado = "SIN_MEMORIA" if oom else "ERROR"
                    print(f"-> {nombre} con lote {lote}: {estado}: {str(e)[:300]}")
                    if not oom:
                        traceback.print_exc()
                    w.writerow([time.strftime("%Y-%m-%d %H:%M"), nombre, a.marco, gpu, lote, estado,
                                "", "", "", "", "", "", "", str(e)[:300].replace("\n", " ")]); f.flush()
                    torch.cuda.empty_cache()
                    if not oom:
                        break  # un error que no es de memoria no se arregla bajando el lote
    print(f"\nTabla: {tabla}")


if __name__ == "__main__":
    main()
