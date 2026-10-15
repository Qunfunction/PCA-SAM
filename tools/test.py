"""Evaluate a full PCA-SAM checkpoint with the configured class-aware COCO metric."""
import argparse
from mmengine.config import DictAction
from mmengine.runner import Runner
from _common import load_config, disable_component_pretraining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--work-dir', default='work_dirs/pca_sam_test')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    cfg = load_config(args.config, args.cfg_options)
    disable_component_pretraining(cfg)
    cfg.load_from = args.checkpoint
    cfg.work_dir = args.work_dir
    Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()
