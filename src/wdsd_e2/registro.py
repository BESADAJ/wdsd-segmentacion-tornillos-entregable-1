"""Registro de modelos del E2 y reglas comunes de presupuesto. Solo biblioteca estándar: lo importa el orquestador.

Reglas (preregistro_E2.md §3 y §4):
- Lote efectivo 2 para todos. Los que no caben con lote 2 en 6 GB usan lote 1 y acumulan 2 iteraciones por paso del
  optimizador (Mask2Former: accumulative_counts=2; YOLOv7/YOLOv9: nbs=2 parchado en segment/train.py).
- Tasa por defecto de cada marco: SGD de MMDetection escalada al lote (oficial x 2/16); AdamW de MMDetection sin escalar;
  ultralytics = lo que elige optimizer=auto con 1 clase y menos de 10 000 pasos (AdamW, 0.002); repos YOLO = lr0 de su
  hyp.scratch-high (SGD, 0.01); Mask R-CNN de torchvision = 0.0025 x lote (receta de torchvision).
- Búsqueda: T = 3 configuraciones, tasa por defecto x {0.5, 1, 2}; el baseline clásico usa λ en {1e-4, 1e-3, 1e-2}.
- Calentamiento lineal de min(500, pasos totales / 10) pasos del optimizador; después coseno hasta el 1 % de la tasa.
"""
from __future__ import annotations

import math

LOTE_EFECTIVO = 2
FACTORES = (0.5, 1.0, 2.0)          # T = 3; el del medio es la configuración por defecto
LAMBDAS = (1e-4, 1e-3, 1e-2)        # baseline clásico; el del medio es el de por defecto
EMPATE = 0.005                       # si la mejor supera a la de por defecto por menos de esto, gana la de por defecto
LR_FINAL = 0.01                      # el coseno termina en el 1 % de la tasa inicial
SEMILLA_BUSQUEDA = 42
SEMILLAS = (42, 43, 44, 45, 46)

D = "https://download.openmmlab.com/mmdetection/"

# nombre -> especificación. grupo: top10 | base (E1) | reserva | baseline
MODELOS: dict[str, dict] = {
    # --- Top 10
    "mask2former": dict(grupo="top10", familia="Q", marco="mmdet", entorno="mmdet", lote=1,
                        config="mask2former/mask2former_r50_8xb2-lsj-50e_coco.py",
                        pesos=D + "v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth"),
    "yolov7": dict(grupo="top10", familia="P", marco="oficial", entorno="yolo9", lote=1, repo="yolov7",
                   cfg="models/segment/yolov7-seg.yaml", pesos="yolov7-seg.pt", patron_ckpt="epoch_{}.pt"),
    "yolov9": dict(grupo="top10", familia="P", marco="oficial", entorno="yolo9", lote=1, repo="yolov9",
                   cfg="models/segment/gelan-c-seg.yaml", pesos="gelan-c-seg.pt", patron_ckpt="epoch{}.pt"),
    "yolo26": dict(grupo="top10", familia="P", marco="ultra", entorno="dl", lote=2, variante="yolo26n-seg"),
    "yolo11": dict(grupo="top10", familia="P", marco="ultra", entorno="dl", lote=2, variante="yolo11n-seg"),
    "cascade": dict(grupo="top10", familia="R", marco="mmdet", entorno="mmdet", lote=2,
                    config="cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py",
                    pesos=D + "v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"),
    "htc": dict(grupo="top10", familia="R", marco="mmdet", entorno="mmdet", lote=2,
                config="htc/htc-without-semantic_r50_fpn_1x_coco.py",
                pesos=D + "v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317-7332cf16.pth"),
    "pointrend": dict(grupo="top10", familia="R", marco="mmdet", entorno="mmdet", lote=2,
                      config="point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py",
                      pesos=D + "v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth"),
    "msrcnn": dict(grupo="top10", familia="R", marco="mmdet", entorno="mmdet", lote=2,
                   config="ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py",
                   pesos=D + "v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth"),
    "condinst": dict(grupo="top10", familia="K", marco="mmdet", entorno="mmdet", lote=2,
                     config="condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py",
                     pesos=D + "v3.0/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance/condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223-4c186406.pth"),
    # --- Modelos base del E1 (reentrenados con el protocolo del E2)
    "maskrcnn": dict(grupo="base", familia="R", marco="torchvision", entorno="dl", lote=2),
    "yolov8": dict(grupo="base", familia="P", marco="ultra", entorno="dl", lote=2, variante="yolov8n-seg"),
    "solov2": dict(grupo="base", familia="K", marco="mmdet", entorno="mmdet", lote=2,
                   config="solov2/solov2_r50_fpn_1x_coco.py",
                   pesos=D + "v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth"),
    # --- Baselines
    "clasico": dict(grupo="baseline", familia="-", marco="clasico", entorno="dl", lote=None),
    "vacio": dict(grupo="baseline", familia="-", marco="vacio", entorno="dl", lote=None),
    # --- Reservas (solo entran si un modelo del Top 10 falla técnicamente en las 5 semillas)
    "queryinst": dict(grupo="reserva", familia="Q", marco="mmdet", entorno="mmdet", lote=2,
                      config="queryinst/queryinst_r50_fpn_1x_coco.py",
                      pesos=D + "v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth"),
    "scnet": dict(grupo="reserva", familia="R", marco="mmdet", entorno="mmdet", lote=2,
                  config="scnet/scnet_r50_fpn_1x_coco.py",
                  pesos=D + "v2.0/scnet/scnet_r50_fpn_1x_coco/scnet_r50_fpn_1x_coco-c3f09857.pth"),
}

TOP10 = [m for m, s in MODELOS.items() if s["grupo"] == "top10"]
BASE = [m for m, s in MODELOS.items() if s["grupo"] == "base"]
ENTRENABLES = TOP10 + BASE + ["clasico"]   # los que pasan por búsqueda y corridas finales ("vacio" no se entrena)


def acumulacion(modelo: str) -> int:
    lote = MODELOS[modelo]["lote"]
    return LOTE_EFECTIVO // lote if lote else 1


def configs(modelo: str) -> tuple:
    """Valores de la búsqueda (T = 3) y el de por defecto."""
    return (LAMBDAS, LAMBDAS[1]) if MODELOS[modelo]["marco"] == "clasico" else (FACTORES, FACTORES[1])


def etiqueta(valor: float) -> str:
    return f"{valor:g}"


def epocas_busqueda(epocas_final: int) -> int:
    return math.ceil(epocas_final / 3)


def pasos_por_epoca(n_train: int) -> int:
    return math.ceil(n_train / LOTE_EFECTIVO)


def calentamiento_pasos(epocas: int, n_train: int) -> int:
    """min(500, pasos totales / 10), en pasos del optimizador (lote efectivo 2); igual a la Mask R-CNN del E1."""
    return min(500, max(1, epocas * pasos_por_epoca(n_train) // 10))


def calentamiento_epocas(epocas: int, n_train: int) -> float:
    return calentamiento_pasos(epocas, n_train) / pasos_por_epoca(n_train)


def resolver(lista: str) -> list[str]:
    """'todos' = Top 10 + base + clásico; 'top10', 'base'; o nombres separados por comas."""
    salida = []
    for x in [t.strip() for t in lista.split(",") if t.strip()]:
        if x == "todos":
            salida += ENTRENABLES
        elif x == "top10":
            salida += TOP10
        elif x == "base":
            salida += BASE
        elif x in MODELOS:
            salida.append(x)
        else:
            raise SystemExit(f"Modelo desconocido: {x}. Opciones: todos, top10, base, {', '.join(MODELOS)}")
    return list(dict.fromkeys(salida))
