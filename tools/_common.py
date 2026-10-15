"""Shared configuration and registration helpers for command-line entry points."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_config(path, overrides=None):
    from mmengine.config import Config
    from mmdet.utils import register_all_modules
    import pca_sam

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(path)
    if overrides:
        cfg.merge_from_dict(overrides)
    return cfg


def disable_component_pretraining(cfg):
    """A full PCA-SAM checkpoint already contains the three SAM components."""
    cfg.model.backbone.checkpoint = None
    cfg.model.roi_head.mask_decoder.checkpoint = None
    cfg.model.roi_head.prompt_encoder.checkpoint = None
