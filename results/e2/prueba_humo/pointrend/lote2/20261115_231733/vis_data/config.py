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
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint='open-mmlab://detectron2/resnet50_caffe',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='caffe',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            fc_out_channels=1024,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=1,
            num_fcs=2,
            type='CoarseMaskHead'),
        mask_roi_extractor=dict(
            aggregation='concat',
            featmap_strides=[
                4,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, type='SimpleRoIAlign'),
            type='GenericRoIExtractor'),
        point_head=dict(
            coarse_pred_each_layer=True,
            fc_channels=256,
            in_channels=256,
            loss_point=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=1,
            num_fcs=3,
            type='MaskPointHead'),
        type='PointRendRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                2,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            scale_factor=2,
            score_thr=0.05,
            subdivision_num_points=784,
            subdivision_steps=5),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            importance_sample_ratio=0.75,
            mask_size=7,
            num_points=196,
            oversample_ratio=3,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='PointRend')
optim_wrapper = dict(
    optimizer=dict(lr=0.0025, momentum=0.9, type='SGD', weight_decay=0.0001),
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
    dict(
        keep_ratio=True,
        scales=[
            (
                1333,
                640,
            ),
            (
                1333,
                672,
            ),
            (
                1333,
                704,
            ),
            (
                1333,
                736,
            ),
            (
                1333,
                768,
            ),
            (
                1333,
                800,
            ),
        ],
        type='RandomChoiceResize'),
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
    name='vis_pointrend_2_1794802653353767800',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'D:\\Documents\\Personal\\Uninorte\\Maestría\\Primer semestre\\Deep Learning\\Trabajo\\datos\\wdsd\\results\\e2\\prueba_humo\\pointrend\\lote2'
