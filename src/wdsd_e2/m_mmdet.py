"""Trabajador MMDetection 3.3 (entorno wdsd_mmdet): Mask2Former, Cascade, HTC, PointRend, MS R-CNN, CondInst,
SOLOv2 y las reservas QueryInst y SCNet.

Parte de la configuración oficial de cada modelo (mmdet/.mim/configs) y cambia lo mismo en todos, como en la prueba de
humo (prueba_humo_e2.py): 1 clase, 1280x720 sin reescalar, volteo horizontal como única aumentación, anclas de 8 a
128 px en los modelos con RPN, sin rama semántica en HTC y SCNet, pesos COCO oficiales. El programa de entrenamiento
pasa a ser por épocas, con el calentamiento y el coseno comunes del E2 (registro.py). Mejor época: AP de máscara en val.
"""
from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

from . import registro as R
from .comun import DATOS, carpeta_imagenes, guardar_preds, imagenes, n_train, pico_gb, ruta_json

ESCALA_ANCLA = 2  # tamaño de ancla = 2 x paso del nivel FPN (4..64) -> 8..128 px (oficial 8 -> 32..512 px)


def _poner(obj, clave, valor):
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k == clave:
                obj[k] = valor
            else:
                _poner(obj[k], clave, valor)
    elif isinstance(obj, list):
        for x in obj:
            _poner(x, clave, valor)


def configuracion(modelo: str, carpeta: Path, semilla: int, epocas: int, factor: float, preentrenado: bool = True,
                  amp: bool = True):
    import mmdet
    from mmengine.config import Config

    spec = R.MODELOS[modelo]
    lote, acum = spec["lote"], R.acumulacion(modelo)
    cfg = Config.fromfile(str(Path(mmdet.__file__).parent / ".mim" / "configs" / spec["config"]))
    _poner(cfg.model, "num_classes", 1)
    if modelo == "mask2former":
        _poner(cfg.model, "num_things_classes", 1)
        _poner(cfg.model, "num_stuff_classes", 0)
        cfg.model.panoptic_head.loss_cls.class_weight = [1.0, 0.1]
        cfg.model.data_preprocessor.batch_augments = None  # sin el relleno fijo a 1024x1024 de LSJ
    if modelo == "scnet":  # sin rama semántica (necesita COCO-stuff), igual que HTC
        cfg.model.roi_head.semantic_roi_extractor = None
        cfg.model.roi_head.semantic_head = None
    rpn = cfg.model.get("rpn_head")
    if rpn is not None and "anchor_generator" in rpn:
        rpn.anchor_generator.scales = [ESCALA_ANCLA]

    meta = dict(classes=("tornillo",))
    tren = [dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
            dict(type="Resize", scale=(1280, 720), keep_ratio=True), dict(type="RandomFlip", prob=0.5),
            dict(type="PackDetInputs")]
    prueba = [dict(type="LoadImageFromFile"), dict(type="Resize", scale=(1280, 720), keep_ratio=True),
              dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"))]

    def ds(split, pipe, test):
        return dict(type="CocoDataset", data_root="", metainfo=meta, ann_file=ruta_json(split).resolve().as_posix(),
                    data_prefix=dict(img=carpeta_imagenes(split).resolve().as_posix() + "/"), test_mode=test,
                    filter_cfg=dict(filter_empty_gt=not test, min_size=0), pipeline=pipe)

    cfg.train_dataloader = dict(batch_size=lote, num_workers=0, persistent_workers=False,
                                sampler=dict(type="DefaultSampler", shuffle=True),
                                batch_sampler=dict(type="AspectRatioBatchSampler"), dataset=ds("train", tren, False))
    cfg.val_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False, drop_last=False,
                              sampler=dict(type="DefaultSampler", shuffle=False), dataset=ds("val", prueba, True))
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator = dict(type="CocoMetric", metric="segm", format_only=False,
                             ann_file=ruta_json("val").resolve().as_posix())
    cfg.test_evaluator = cfg.val_evaluator
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=epocas, val_interval=1)
    cfg.val_cfg = dict(type="ValLoop")
    cfg.test_cfg = dict(type="TestLoop")

    # Optimizador oficial. SGD: tasa oficial (lote 16) escalada al lote efectivo 2. AdamW: la oficial, sin escalar.
    opt = cfg.optim_wrapper.optimizer
    lr_def = opt.lr * R.LOTE_EFECTIVO / 16 if opt.type == "SGD" else opt.lr
    opt.lr = lr_def * factor
    cfg.optim_wrapper.type = "AmpOptimWrapper" if amp else "OptimWrapper"  # AMP solo existe en GPU
    cfg.optim_wrapper.accumulative_counts = acum
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)

    # Programa común en iteraciones: iteraciones = pasos x acumulación.
    it_epoca = math.ceil(n_train(con_anotaciones=True) / lote)  # filter_empty_gt quita las imágenes sin tornillos
    total = epocas * it_epoca
    w = max(2, min(R.calentamiento_pasos(epocas, n_train(con_anotaciones=True)) * acum, total - 1))
    cfg.param_scheduler = [
        dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=w),
        dict(type="CosineAnnealingLR", by_epoch=False, begin=w, end=total, T_max=total - w, eta_min_ratio=R.LR_FINAL)]
    cfg.default_hooks = dict(
        timer=dict(type="IterTimerHook"), logger=dict(type="LoggerHook", interval=50),
        param_scheduler=dict(type="ParamSchedulerHook"),
        checkpoint=dict(type="CheckpointHook", interval=1, by_epoch=True, max_keep_ckpts=1, save_last=True,
                        save_best="coco/segm_mAP", rule="greater"),
        sampler_seed=dict(type="DistSamplerSeedHook"))
    cfg.log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
    cfg.load_from = spec["pesos"] if preentrenado else None
    cfg.resume = False
    cfg.randomness = dict(seed=semilla, deterministic=False)
    cfg.env_cfg = dict(cudnn_benchmark=False, dist_cfg=dict(backend="gloo" if platform.system() == "Windows" else "nccl"))
    cfg.work_dir = str(carpeta.resolve())
    cfg.visualizer = dict(type="DetLocalVisualizer", vis_backends=[dict(type="LocalVisBackend")],
                          name=f"vis_{modelo}_{semilla}_{time.time_ns()}")
    if not preentrenado and "init_cfg" in cfg.model.get("backbone", {}):
        cfg.model.backbone.init_cfg = None  # solo pruebas sin internet
    info = {"lr_inicial": opt.lr, "lr_por_defecto": lr_def, "optimizador": opt.type, "lote": lote, "acumulacion": acum,
            "iter_por_epoca": it_epoca, "calentamiento_iter": w, "iter_totales": total}
    return cfg, info


def historial(carpeta: Path) -> list[dict]:
    """Pérdida media de train y AP de máscara en val por época, desde el registro JSON de mmengine."""
    archivos = sorted(carpeta.glob("*/vis_data/scalars.json")) or sorted(carpeta.glob("*/vis_data/*.json"))
    if not archivos:
        return []
    perdida, val = {}, {}
    for linea in archivos[-1].read_text(encoding="utf8").splitlines():
        d = json.loads(linea)
        if "coco/segm_mAP" in d:
            val[int(d["step"])] = d
        elif "loss" in d and "epoch" in d:
            perdida.setdefault(int(d["epoch"]), []).append(float(d["loss"]))
    filas = []
    for ep in sorted(set(perdida) | set(val)):
        v = val.get(ep, {})
        filas.append({"epoca": ep, "perdida": sum(perdida.get(ep, [math.nan])) / max(len(perdida.get(ep, [])), 1),
                      "val_AP": v.get("coco/segm_mAP"), "val_AP50": v.get("coco/segm_mAP_50"),
                      "val_AP75": v.get("coco/segm_mAP_75"), "val_AP_small": v.get("coco/segm_mAP_s")})
    return filas


def predecir(detector, split: str) -> list[dict]:
    from mmdet.apis import inference_detector
    from wdsd_inst.metricas import a_rle
    salida = []
    for im in imagenes(split):
        ruta = carpeta_imagenes(split) / f"{Path(im['file_name']).stem}.png"
        r = inference_detector(detector, str(ruta)).pred_instances
        ms, sc = r.masks.cpu().numpy(), r.scores.float().cpu().numpy()
        for s, m in zip(sc, ms):
            if m.shape != (im["height"], im["width"]):
                raise ValueError(f"Máscara de forma {m.shape}, esperaba {(im['height'], im['width'])}")
            if m.any():
                salida.append({"image_id": im["id"], "category_id": 1, "segmentation": a_rle(m), "score": float(s)})
    return salida


def entrenar(modelo: str, carpeta: Path, semilla: int, epocas: int, factor: float, splits: tuple, disp,
             preentrenado: bool = True, conservar_pesos: bool = True, log=print) -> dict:
    import torch
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules
    from mmengine.runner import Runner

    register_all_modules()
    cfg, info = configuracion(modelo, carpeta, semilla, epocas, factor, preentrenado, amp=disp.type == "cuda")
    log(f"  {modelo}: {info}")
    reiniciar = torch.cuda.reset_peak_memory_stats
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None  # mmengine lo llama en cada log; queremos el pico total
    t0 = time.time()
    try:
        Runner.from_cfg(cfg).train()
    finally:
        torch.cuda.reset_peak_memory_stats = reiniciar
    t_entrenar = time.time() - t0
    memoria = pico_gb()
    mejores = sorted(carpeta.glob("best_coco_segm_mAP_epoch_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    # Si ninguna época dio detecciones en val, CocoMetric no devuelve coco/segm_mAP y no hay "best": todas valen AP 0
    # y se usa la última (como en el E1 con SOLOv2).
    sin_best = not mejores
    mejores = mejores or sorted(carpeta.glob("epoch_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not mejores:
        raise RuntimeError("mmengine no guardó ningún checkpoint")
    mejor = mejores[-1]
    t1 = time.time()
    detector = init_detector(cfg, str(mejor), device=str(disp))
    for split in splits:
        guardar_preds(carpeta, split, predecir(detector, split))
    t_predecir = time.time() - t1
    params = sum(p.numel() for p in detector.parameters())
    for p in carpeta.glob("*.pth"):  # se queda solo el mejor checkpoint (o ninguno)
        if p != mejor or not conservar_pesos:
            p.unlink()
    if conservar_pesos:  # sin el estado del optimizador: la mitad del tamaño
        ck = torch.load(mejor, map_location="cpu")
        torch.save({"meta": ck.get("meta"), "state_dict": ck["state_dict"]}, carpeta / "mejor.pth")
        mejor.unlink()
        del ck
    for p in carpeta.glob("last_checkpoint"):
        p.unlink()
    return {**info, **memoria, "t_entrenar_s": t_entrenar, "t_predecir_s": t_predecir,
            "mejor_epoca": int(mejor.stem.rsplit("_", 1)[1]), "val_sin_detecciones": sin_best, "parametros": params,
            "historial": historial(carpeta)}
