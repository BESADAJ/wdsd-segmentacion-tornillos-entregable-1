"""SOLOv2 (MMDetection 3.3, ResNet-50-FPN preentrenado en COCO) con el mismo protocolo y la misma evaluación.

Va en un entorno aparte (mmcv necesita torch 2.1). Ver ENTRENAMIENTO_INSTANCIAS.md, sección SOLOv2.

  python solov2_instancias.py piloto
  python solov2_instancias.py cola --hasta 07:30 --epocas 12 --semillas 42,43,44,45,46

Requiere haber corrido antes `entrenar_instancias.py preparar` (usa datos_inst/coco/*.json y las imágenes
enlazadas en datos_inst/yolo/images/<split>/). Deja en results/instancias/corridas/solov2_s<semilla>/ los
mismos archivos que los otros modelos (preds_*.json, metricas.json, test_por_imagen.csv, resumen.json).

Configuración: la oficial solov2_r50_fpn_1x_coco de MMDetection (rejillas 40/36/24/16/12, Dice con peso 3,
focal gamma 2 alpha 0.25), con 1 clase, sin reescalar (1280x720), volteo horizontal, lr lineal al lote
(0.01 para lote 16), mejor época por AP de máscara en val.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
DATOS = RAIZ / "datos_inst"
SALIDA = RAIZ / "results" / "instancias"
PESOS_COCO = ("https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/"
              "solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth")


def configuracion(carpeta: Path, semilla: int, epocas: int, lote: int, amp: bool, preentrenado: bool):
    from mmengine.config import Config

    def img_dir(s):
        return (DATOS / "yolo" / "images" / s).resolve().as_posix() + "/"

    meta = dict(classes=("tornillo",))
    tren = [dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
            dict(type="Resize", scale=(1280, 720), keep_ratio=True),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PackDetInputs")]
    prueba = [dict(type="LoadImageFromFile"),
              dict(type="Resize", scale=(1280, 720), keep_ratio=True),
              dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"))]

    def cargador(split, pipeline, entrenar):
        return dict(batch_size=lote if entrenar else 1, num_workers=0, persistent_workers=False,
                    sampler=dict(type="DefaultSampler", shuffle=entrenar),
                    batch_sampler=dict(type="AspectRatioBatchSampler") if entrenar else None,
                    dataset=dict(type="CocoDataset", data_root="", metainfo=meta,
                                 ann_file=(DATOS / "coco" / f"{split}.json").resolve().as_posix(),
                                 data_prefix=dict(img=img_dir(split)), test_mode=not entrenar,
                                 filter_cfg=dict(filter_empty_gt=entrenar, min_size=0), pipeline=pipeline))

    evaluador = lambda split: dict(type="CocoMetric", metric="segm", format_only=False,
                                   ann_file=(DATOS / "coco" / f"{split}.json").resolve().as_posix())
    cfg = dict(
        default_scope="mmdet",
        model=dict(
            type="SOLOv2",
            data_preprocessor=dict(type="DetDataPreprocessor", mean=[123.675, 116.28, 103.53],
                                   std=[58.395, 57.12, 57.375], bgr_to_rgb=True, pad_mask=True, pad_size_divisor=32),
            backbone=dict(type="ResNet", depth=50, num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1,
                          init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50") if preentrenado else None,
                          style="pytorch"),
            neck=dict(type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, start_level=0, num_outs=5),
            mask_head=dict(
                type="SOLOV2Head", num_classes=1, in_channels=256, feat_channels=512, stacked_convs=4,
                strides=[8, 8, 16, 32, 32], scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                pos_scale=0.2, num_grids=[40, 36, 24, 16, 12], cls_down_index=0,
                mask_feature_head=dict(feat_channels=128, start_level=0, end_level=3, out_channels=256, mask_stride=4,
                                       norm_cfg=dict(type="GN", num_groups=32, requires_grad=True)),
                loss_mask=dict(type="DiceLoss", use_sigmoid=True, loss_weight=3.0),
                loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)),
            test_cfg=dict(nms_pre=500, score_thr=0.05, mask_thr=0.5, filter_thr=0.05, kernel="gaussian", sigma=2.0,
                          max_per_img=100)),
        train_dataloader=cargador("train", tren, True),
        val_dataloader=cargador("val", prueba, False),
        test_dataloader=cargador("test", prueba, False),
        val_evaluator=evaluador("val"), test_evaluator=evaluador("test"),
        train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=epocas, val_interval=1),
        val_cfg=dict(type="ValLoop"), test_cfg=dict(type="TestLoop"),
        optim_wrapper=dict(type="AmpOptimWrapper" if amp else "OptimWrapper",
                           optimizer=dict(type="SGD", lr=0.01 * lote / 16, momentum=0.9, weight_decay=1e-4),
                           clip_grad=dict(max_norm=35, norm_type=2)),
        param_scheduler=[dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=300),
                         dict(type="CosineAnnealingLR", by_epoch=True, begin=0, end=epocas, T_max=epocas)],
        default_hooks=dict(timer=dict(type="IterTimerHook"), logger=dict(type="LoggerHook", interval=50),
                           param_scheduler=dict(type="ParamSchedulerHook"),
                           checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1,
                                           save_best="coco/segm_mAP", rule="greater"),
                           sampler_seed=dict(type="DistSamplerSeedHook")),
        env_cfg=dict(cudnn_benchmark=False, dist_cfg=dict(backend="gloo" if platform.system() == "Windows" else "nccl")),
        log_level="INFO", load_from=PESOS_COCO if preentrenado else None, resume=False,
        randomness=dict(seed=semilla, deterministic=False),
        work_dir=str(carpeta.resolve()),
    )
    return Config(cfg)


def predecir(modelo, split: str) -> list[dict]:
    from mmdet.apis import inference_detector
    from wdsd_inst.metricas import a_rle
    gt = json.loads((DATOS / "coco" / f"{split}.json").read_text(encoding="utf8"))
    salida = []
    for im in gt["images"]:
        ruta = DATOS / "yolo" / "images" / split / f"{Path(im['file_name']).stem}.png"
        r = inference_detector(modelo, str(ruta)).pred_instances
        ms, sc = r.masks.cpu().numpy(), r.scores.cpu().numpy()
        for s, m in zip(sc, ms):
            if m.shape != (im["height"], im["width"]):
                raise ValueError(f"Máscara de forma {m.shape}, esperaba {(im['height'], im['width'])}")
            if m.any():
                salida.append({"image_id": im["id"], "category_id": 1, "segmentation": a_rle(m), "score": float(s)})
    return salida


def correr(semilla: int, epocas: int, carpeta: Path, args) -> dict:
    import torch
    from mmengine.runner import Runner
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules
    from wdsd_inst.metricas import evaluar_corrida

    if not torch.cuda.is_available() and not args.permitir_cpu:
        sys.exit(f"ERROR: torch {torch.__version__} no ve la GPU (torch.cuda.is_available() = False). "
                 "Entrenar en CPU tardaría horas. Revisa la instalación o usa --permitir-cpu solo para pruebas.")
    register_all_modules()
    carpeta.mkdir(parents=True, exist_ok=True)
    cfg = configuracion(carpeta, semilla, epocas, args.lote, not args.sin_amp, not args.sin_preentrenar)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    Runner.from_cfg(cfg).train()
    t_train = time.time() - t0
    pico = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
    mejores = sorted(carpeta.glob("best_coco_segm_mAP_epoch_*.pth")) or sorted(carpeta.glob("epoch_*.pth"))
    disp = "cuda:0" if torch.cuda.is_available() else "cpu"
    modelo = init_detector(cfg, str(mejores[-1]), device=disp)
    for split in ("val", "test"):
        (carpeta / f"preds_{split}.json").write_text(json.dumps(predecir(modelo, split)))
    m = evaluar_corrida(carpeta, DATOS / "coco/val.json", DATOS / "coco/test.json")
    import mmcv, mmdet
    resumen = {"modelo": "solov2", "semilla": semilla, "epocas": epocas, "s_total": time.time() - t0,
               "s_por_epoca": t_train / epocas, "gpu_pico_gb": pico, "mejor_checkpoint": mejores[-1].name,
               "parametros": sum(p.numel() for p in modelo.parameters()),
               "entorno": {"python": platform.python_version(), "torch": torch.__version__, "mmcv": mmcv.__version__,
                           "mmdet": mmdet.__version__, "so": platform.platform(),
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
               "metricas": m}
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=1, default=str), encoding="utf8")
    print(f"== solov2 s{semilla}: test F1 img {m['test']['F1_img']:.3f} | AP {m['test']['AP']:.3f} | PQ img {m['test']['PQ_img']:.3f} | "
          f"PQ grupo {m['test']['PQ_grupo']:.3f} | {resumen['s_total'] / 60:.1f} min")
    return resumen


def cmd_piloto(args):
    r = correr(42, 1, SALIDA / "piloto" / "solov2", args)
    print(f"\nPiloto SOLOv2: {r['s_total']:.0f} s la corrida de 1 época (entrenamiento {r['s_por_epoca']:.0f} s), "
          f"memoria pico {r['gpu_pico_gb']:.2f} GB, {r['parametros']:,} parámetros.")
    for e in (6, 8, 12, 16, 24):
        h = 5 * (r["s_por_epoca"] * e + (r["s_total"] - r["s_por_epoca"])) / 3600
        print(f"  {e:3d} épocas x 5 semillas -> {h:5.1f} h")


def cmd_cola(args):
    hh, mm = map(int, args.hasta.split(":"))
    ahora = dt.datetime.now()
    lim = ahora.replace(hour=hh, minute=mm, second=0, microsecond=0)
    lim = lim if lim > ahora else lim + dt.timedelta(days=1)
    piloto = SALIDA / "piloto" / "solov2" / "resumen.json"
    s_ep = json.loads(piloto.read_text())["s_por_epoca"] if piloto.exists() else None
    for s in [int(x) for x in args.semillas.split(",")]:
        carpeta = SALIDA / "corridas" / f"solov2_s{s}"
        if (carpeta / "resumen.json").exists():
            print(f"Salto solov2 s{s}: ya terminó."); continue
        if s_ep and dt.datetime.now() + dt.timedelta(seconds=s_ep * args.epocas * 1.1 + 180) > lim:
            print(f"Me detengo: solov2 s{s} no termina antes de {lim:%H:%M}."); return
        print(f"\n== {dt.datetime.now():%H:%M} empiezo solov2 s{s} ({args.epocas} épocas)")
        correr(s, args.epocas, carpeta, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre, f in (("piloto", cmd_piloto), ("cola", cmd_cola)):
        p = sub.add_parser(nombre)
        p.add_argument("--lote", type=int, default=2)
        p.add_argument("--sin-amp", action="store_true")
        p.add_argument("--sin-preentrenar", action="store_true", help="solo para pruebas sin internet")
        p.add_argument("--permitir-cpu", action="store_true", help="solo para pruebas: deja entrenar sin GPU")
        p.set_defaults(f=f)
        if nombre == "cola":
            p.add_argument("--hasta", required=True)
            p.add_argument("--epocas", type=int, default=12)
            p.add_argument("--semillas", default="42,43,44,45,46")
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
