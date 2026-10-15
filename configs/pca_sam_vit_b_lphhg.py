"""PCA-SAM LP-HHG training and class-aware COCO evaluation."""
_base_ = ['./_base_/pca_sam_vit_b.py']
import os as _os

custom_imports = dict(imports=['pca_sam'], allow_failed_imports=False)
default_scope = 'mmdet'

data_root = _os.getenv('PCA_SAM_DATA_ROOT', 'data/LP-HHG/')
checkpoint_root = _os.getenv('PCA_SAM_SAM_ROOT', 'checkpoints/sam_vit_b/')
metainfo = dict(classes=('normal_tumor', 'immune'),
                palette=[(48, 164, 220), (255, 212, 64)])
model = dict(
    backbone=dict(checkpoint=_os.path.join(checkpoint_root, 'image_encoder_vit_b.pth')),
    roi_head=dict(
        mask_decoder=dict(checkpoint=_os.path.join(checkpoint_root, 'mask_decoder_vit_b.pth')),
        prompt_encoder=dict(checkpoint=_os.path.join(checkpoint_root, 'prompt_encoder_vit_b.pth'))))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
train_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='CocoDataset', data_root=data_root,
                 ann_file='annotations/instances_train.json',
                 data_prefix=dict(img='images/'), metainfo=metainfo,
                 filter_cfg=dict(filter_empty_gt=True, min_size=1),
                 pipeline=train_pipeline))
test_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root=data_root,
                 ann_file='annotations/instances_test.json',
                 data_prefix=dict(img='images/'), metainfo=metainfo,
                 test_mode=True, pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric', ann_file=_os.path.join(data_root, 'annotations/instances_test.json'),
    metric=['bbox', 'segm'], classwise=True, proposal_nums=(100, 300, 1000))
test_cfg = dict(type='TestLoop')
val_cfg = None
val_dataloader = None
val_evaluator = None
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=13)
optim_wrapper = dict(
    type='AmpOptimWrapper', accumulative_counts=2,
    loss_scale=dict(init_scale=512, growth_interval=2000),
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True,
         milestones=[8, 11], gamma=0.1),
]
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
               dist_cfg=dict(backend='gloo' if _os.name == 'nt' else 'nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
randomness = dict(seed=20261011, deterministic=False)
load_from = None
resume = False
work_dir = 'work_dirs/pca_sam_lphhg'
