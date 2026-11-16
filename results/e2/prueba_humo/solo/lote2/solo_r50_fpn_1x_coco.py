auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1000000000, save_last=False, type='CheckpointHook'),
    logger=dict(interval=25, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(cudnn_benchmark=False, dist_cfg=dict(backend='gloo'))
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/solo/solo_r50_fpn_1x_coco/solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    mask_head=dict(
        cls_down_index=0,
        feat_channels=256,
        in_channels=256,
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_mask=dict(loss_weight=3.0, type='DiceLoss', use_sigmoid=True),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=1,
        num_grids=[
            40,
            36,
            24,
            16,
            12,
        ],
        pos_scale=0.2,
        scale_ranges=(
            (
                1,
                96,
            ),
            (
                48,
                192,
            ),
            (
                96,
                384,
            ),
            (
                192,
                768,
            ),
            (
                384,
                2048,
            ),
        ),
        stacked_convs=7,
        strides=[
            8,
            8,
            16,
            32,
            32,
        ],
        type='SOLOHead'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=0,
        type='FPN'),
    test_cfg=dict(
        filter_thr=0.05,
        kernel='gaussian',
        mask_thr=0.5,
        max_per_img=100,
        nms_pre=500,
        score_thr=0.1,
        sigma=2.0),
    type='SOLO')
optim_wrapper = dict(
    optimizer=dict(lr=0.00125, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=20, start_factor=0.1, type='LinearLR'),
]
randomness = dict(deterministic=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'D:/Documents/Personal/Uninorte/Maestría/Primer semestre/Deep Learning/Trabajo/datos/wdsd/datos_inst/coco/humo4.json',
        data_prefix=dict(
            img=
            'D:/Documents/Personal/Uninorte/Maestría/Primer semestre/Deep Learning/Trabajo/datos/wdsd/datos_inst/yolo/images/train/'
        ),
        data_root='',
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=dict(classes=('tornillo', )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1280,
                720,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    'D:/Documents/Personal/Uninorte/Maestría/Primer semestre/Deep Learning/Trabajo/datos/wdsd/datos_inst/coco/humo4.json',
    format_only=False,
    metric='segm',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_iters=300, type='IterBasedTrainLoop', val_interval=300)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        'D:/Documents/Personal/Uninorte/Maestría/Primer semestre/Deep Learning/Trabajo/datos/wdsd/datos_inst/coco/humo4.json',
        data_prefix=dict(
            img=
            'D:/Documents/Personal/Uninorte/Maestría/Primer semestre/Deep Learning/Trabajo/datos/wdsd/datos_inst/yolo/images/train/'
        ),
        data_root='',
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        metainfo=dict(classes=('tornillo', )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(keep_ratio=True, scale=(
                1280,
                720,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        test_mode=False,
        type='CocoDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'D:/Documents/Personal/Uninorte/Maestría/Primer semestre/Deep Learning/Trabajo/datos/wdsd/datos_inst/coco/humo4.json',
        data_prefix=dict(
            img=
            'D:/Documents/Personal/Uninorte/Maestría/Primer semestre/Deep Learning/Trabajo/datos/wdsd/datos_inst/yolo/images/train/'
        ),
        data_root='',
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=dict(classes=('tornillo', )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1280,
                720,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    'D:/Documents/Personal/Uninorte/Maestría/Primer semestre/Deep Learning/Trabajo/datos/wdsd/datos_inst/coco/humo4.json',
    format_only=False,
    metric='segm',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'D:\\Documents\\Personal\\Uninorte\\Maestría\\Primer semestre\\Deep Learning\\Trabajo\\datos\\wdsd\\results\\e2\\prueba_humo\\solo\\lote2'
