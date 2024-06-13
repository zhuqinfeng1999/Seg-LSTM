#
# # Optimizer configuration remains the same
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
#
# # Learning policy adjusted for epoch-based training
# param_scheduler = [
#     dict(
#         type='PolyLR',  # Assuming you continue with PolyLR but adjust for epochs
#         eta_min=1e-4,
#         power=0.9,
#         begin=0,  # Specify the max epochs here
#         end=1500,
#         by_epoch=False # Ensure the scheduler operates on an epoch basis
#     )
# ]
#
# # Training schedule adjusted for 50 epochs
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)  # Validate after each epoch
# train_cfg = dict(
#     type='IterBasedTrainLoop', max_iters=29350, val_interval=587)#1000 output val once
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
#
# # default_hooks = dict(
# #     timer=dict(type='EpochTimerHook'),  # Adjusted to EpochTimerHook for epoch-based timing
# #     logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),  # Log every 10 epochs
# #     param_scheduler=dict(type='ParamSchedulerHook'),
# #     checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1),  # Save checkpoint every epoch
# #     sampler_seed=dict(type='DistSamplerSeedHook'),
# #     visualization=dict(type='SegVisualizationHook')
# # )
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=29350),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     #visualization=dict(type='SegVisualizationHook'))
#     visualization=dict(type='SegVisualizationHook',draw=True,interval=50))#results interval=50 it can be set to 1 if want to output every results
#
#
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
        end=40000,
        by_epoch=False)
]
# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=587)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
