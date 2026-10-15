"""Export boxes, two-class labels and RLE masks from a single HHG image."""
import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from mmdet.apis import init_detector, inference_detector
from _common import load_config, disable_component_pretraining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--visualization')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if args.visualization and Path(args.visualization).exists():
        raise FileExistsError(f'Refusing to overwrite {args.visualization}')
    cfg = load_config(args.config)
    disable_component_pretraining(cfg)
    cfg.test_pipeline = [step for step in cfg.test_pipeline
                         if step.type != 'LoadAnnotations']
    cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline
    model = init_detector(cfg, args.checkpoint, device=args.device)
    result = inference_detector(model, args.image)
    predictions = result.pred_instances.cpu()
    names = cfg.metainfo.classes
    records = []
    for box, label, score, mask in zip(predictions.bboxes.numpy(),
                                      predictions.labels.numpy(),
                                      predictions.scores.numpy(),
                                      predictions.masks.numpy()):
        if float(score) < args.score_thr:
            continue
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        records.append(dict(bbox_xyxy=box.tolist(), label=int(label),
                            category_id=int(label) + 1, class_name=names[int(label)],
                            score=float(score), segmentation=rle))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(dict(image=Path(args.image).name, instances=records), stream, indent=2)
    if args.visualization:
        import mmcv
        from mmdet.visualization import DetLocalVisualizer
        path = Path(args.visualization)
        path.parent.mkdir(parents=True, exist_ok=True)
        vis = DetLocalVisualizer()
        vis.dataset_meta = model.dataset_meta
        vis.add_datasample('prediction', mmcv.imread(args.image, channel_order='rgb'),
                           data_sample=result, draw_gt=False,
                           pred_score_thr=args.score_thr, out_file=str(path))
    print(f'Exported {len(records)} instances to {output}')


if __name__ == '__main__':
    main()
