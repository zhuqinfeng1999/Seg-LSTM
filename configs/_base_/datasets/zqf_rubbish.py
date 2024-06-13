# dataset settings
dataset_type = 'RubbishDataset'
data_root = '/ZQF/project/mmseg121/mmsegmentation/data/rubbish_minitest'
crop_size = (512, 512)
img_scale = (1024, 768)
train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    # dict(type='Resize', scale=img_scale, keep_ratio=True),  # Adjust to resize to 1024x1024
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=60),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadTiffImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img/train', seg_map_path='ann/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img/val', seg_map_path='ann/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
