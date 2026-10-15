"""Split an official SAM ViT-B checkpoint into the PCA-SAM initialization files."""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Official sam_vit_b_01ec64.pth')
    parser.add_argument('--output-dir', default='checkpoints/sam_vit_b')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    components = ['image_encoder', 'mask_decoder', 'prompt_encoder']
    outputs = {name: Path(args.output_dir) / f'{name}_vit_b.pth' for name in components}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError('Output components already exist; use a new output directory.')
    subsets = {name: {key[len(name) + 1:]: value for key, value in state.items()
                      if key.startswith(name + '.')} for name in components}
    if not all(subsets.values()):
        raise ValueError('Checkpoint does not contain all three SAM components.')
    if subsets['image_encoder'].get('pos_embed', torch.empty(0)).shape != (1, 64, 64, 768):
        raise ValueError('Expected the official SAM ViT-B image encoder.')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for name, path in outputs.items():
        torch.save(subsets[name], path)
        print(path)


if __name__ == '__main__':
    main()
