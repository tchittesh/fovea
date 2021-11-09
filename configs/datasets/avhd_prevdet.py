dataset_type = 'AVHDPrevDetDataset'
original_image_shape = (1200, 1920)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Timer', name='preprocessing', transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            ]),
            dict(
                type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
                meta_keys=[
                    'filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg', 'preprocessing_time']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='Argoverse-HD/annotations/htc_dconv2_ms_train.json',
        img_prefix='Argoverse-1.1/tracking',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='Argoverse-HD/annotations/val.json',
        img_prefix='Argoverse-1.1/tracking',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='Argoverse-HD/annotations/val.json',
        img_prefix='Argoverse-1.1/tracking',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'], classwise=True)
