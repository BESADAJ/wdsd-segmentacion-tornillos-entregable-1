default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        rule='greater',
        save_best='coco/segm_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmdet'
env_cfg = dict(cudnn_benchmark=False, dist_cfg=dict(backend='gloo'))
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth'
log_level = 'INFO'
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
        feat_channels=512,
        in_channels=256,
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_mask=dict(loss_weight=3.0, type='DiceLoss', use_sigmoid=True),
        mask_feature_head=dict(
            end_level=3,
            feat_channels=128,
            mask_stride=4,
            norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
            out_channels=256,
            start_level=0),
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
        stacked_convs=4,
        strides=[
            8,
            8,
            16,
            32,
            32,
        ],
        type='SOLOV2Head'),
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
        score_thr=0.05,
        sigma=2.0),
    type='SOLOv2')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.00125, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=300, start_factor=0.001, type='LinearLR'),
    dict(T_max=12, begin=0, by_epoch=True, end=12, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=46)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_sampler=None,
    batch_size=1,
    dataset=dict(
        ann_file=
        'C:/Users/Usuario/Desktop/MachineLearningConda/DEEP_LEARNING/Entregable _1_Deep_Learning/wdsd-entregable-1/datos_inst/coco/test.json',
        data_prefix=dict(
            img=
            'C:/Users/Usuario/Desktop/MachineLearningConda/DEEP_LEARNING/Entregable _1_Deep_Learning/wdsd-entregable-1/datos_inst/yolo/images/test/'
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
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    'C:/Users/Usuario/Desktop/MachineLearningConda/DEEP_LEARNING/Entregable _1_Deep_Learning/wdsd-entregable-1/datos_inst/coco/test.json',
    format_only=False,
    metric='segm',
    type='CocoMetric')
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file=
        'C:/Users/Usuario/Desktop/MachineLearningConda/DEEP_LEARNING/Entregable _1_Deep_Learning/wdsd-entregable-1/datos_inst/coco/train.json',
        data_prefix=dict(
            img=
            'C:/Users/Usuario/Desktop/MachineLearningConda/DEEP_LEARNING/Entregable _1_Deep_Learning/wdsd-entregable-1/datos_inst/yolo/images/train/'
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
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        test_mode=False,
        type='CocoDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_sampler=None,
    batch_size=1,
    dataset=dict(
        ann_file=
        'C:/Users/Usuario/Desktop/MachineLearningConda/DEEP_LEARNING/Entregable _1_Deep_Learning/wdsd-entregable-1/datos_inst/coco/val.json',
        data_prefix=dict(
            img=
            'C:/Users/Usuario/Desktop/MachineLearningConda/DEEP_LEARNING/Entregable _1_Deep_Learning/wdsd-entregable-1/datos_inst/yolo/images/val/'
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
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    'C:/Users/Usuario/Desktop/MachineLearningConda/DEEP_LEARNING/Entregable _1_Deep_Learning/wdsd-entregable-1/datos_inst/coco/val.json',
    format_only=False,
    metric='segm',
    type='CocoMetric')
work_dir = 'C:\\Users\\Usuario\\Desktop\\MachineLearningConda\\DEEP_LEARNING\\Entregable _1_Deep_Learning\\wdsd-entregable-1\\results\\instancias\\corridas\\solov2_s46'
