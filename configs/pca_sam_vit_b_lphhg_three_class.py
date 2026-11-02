"""LP-HHG manuscript configuration: tumor, immune, and normal cells.

Use three-class annotations and a matching checkpoint. This configuration does
not convert the labels in an existing two-class COCO annotation file.
"""
_base_ = ['./pca_sam_vit_b_lphhg.py']

metainfo = dict(classes=('tumor', 'immune', 'normal'),
                palette=[(48, 164, 220), (255, 0, 0), (255, 255, 255)])

# Keep the stage-specific box coders and losses, changing only the class count.
model = dict(roi_head=dict(
    bbox_head=[{'bbox_coder': {'target_means': [0.0, 0.0, 0.0, 0.0],
                 'target_stds': [0.1, 0.1, 0.2, 0.2],
                 'type': 'DeltaXYWHBBoxCoder'},
  'fc_out_channels': 1024,
  'in_channels': 256,
  'loss_bbox': {'beta': 1.0, 'loss_weight': 1.0, 'type': 'SmoothL1Loss'},
  'loss_cls': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_sigmoid': False},
  'num_classes': 3,
  'reg_class_agnostic': True,
  'roi_feat_size': 7,
  'type': 'MTIHContextBBoxHead'},
 {'bbox_coder': {'target_means': [0.0, 0.0, 0.0, 0.0],
                 'target_stds': [0.05, 0.05, 0.1, 0.1],
                 'type': 'DeltaXYWHBBoxCoder'},
  'fc_out_channels': 1024,
  'in_channels': 256,
  'loss_bbox': {'beta': 1.0, 'loss_weight': 1.0, 'type': 'SmoothL1Loss'},
  'loss_cls': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_sigmoid': False},
  'num_classes': 3,
  'reg_class_agnostic': True,
  'roi_feat_size': 7,
  'type': 'MTIHContextBBoxHead'},
 {'bbox_coder': {'target_means': [0.0, 0.0, 0.0, 0.0],
                 'target_stds': [0.033, 0.033, 0.067, 0.067],
                 'type': 'DeltaXYWHBBoxCoder'},
  'fc_out_channels': 1024,
  'in_channels': 256,
  'loss_bbox': {'beta': 1.0, 'loss_weight': 1.0, 'type': 'SmoothL1Loss'},
  'loss_cls': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_sigmoid': False},
  'num_classes': 3,
  'reg_class_agnostic': True,
  'roi_feat_size': 7,
  'type': 'MTIHContextBBoxHead'}],
    mask_head=dict(num_classes=3),
    sam_mask_head=dict(num_classes=3)))
train_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = dict(dataset=dict(metainfo=metainfo))
work_dir = 'work_dirs/pca_sam_lphhg_three_class'
