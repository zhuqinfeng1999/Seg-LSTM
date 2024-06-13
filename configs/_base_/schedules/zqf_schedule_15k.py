# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=15000,
        by_epoch=False)
]
# training schedule for 15k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=15000, val_interval=1000)#1000 output val once
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=15000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    #visualization=dict(type='SegVisualizationHook'))
    visualization=dict(type='SegVisualizationHook',draw=True,interval=50))#results interval=50 it can be set to 1 if want to output every results
