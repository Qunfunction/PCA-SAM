"""Train PCA-SAM with an explicit seed and output directory."""
import argparse
from mmengine.config import DictAction
from mmengine.runner import Runner
from _common import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--work-dir')
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    cfg = load_config(args.config, args.cfg_options)
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    cfg.work_dir = args.work_dir or f'work_dirs/pca_sam_seed_{args.seed}'
    cfg.resume = args.resume
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
