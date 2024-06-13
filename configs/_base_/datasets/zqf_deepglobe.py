# dataset settings
dataset_type = 'DeepglobeDataset'
data_root = 'data/deepglobe'
img_scale = (512, 512)  # New: Set original image scale


train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),  # Adjust to resize to 1024x1024
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),  # Keep for horizontal flip
    dict(type='RandomFlip', prob=0.5, direction='vertical'),  # Add for vertical flip
    # Implement or ensure support for RandomDiagonalFlip
    dict(type='RandomDiagonalFlip', prob=0.5),  # This needs to be implemented or ensured it's supported
    dict(type='PackSegInputs'),
    # dict(type='ConvertToGrayscale'),  # 添加这个步骤
]
test_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),  # Resize to crop size for testing
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
    # dict(type='ConvertToGrayscale'),  # 添加这个步骤
]

# No changes to img_ratios and tta_pipeline for now as they might require advanced handling for diagonal flips

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    # dataset=dict(
    #     # type='RepeatDataset',
    #     # times=40000,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/gt2'),
        pipeline=train_pipeline))
# )

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/gt2'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# Modify metrics to use miou, P, R, F1
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mFscore'])
test_evaluator = val_evaluator
