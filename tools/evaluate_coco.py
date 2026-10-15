"""Evaluate COCO detections with explicit IoU type, category policy and cap."""
import argparse
import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations')
    parser.add_argument('predictions', help='COCO-format result list, not predict.py export JSON')
    parser.add_argument('--iou-type', choices=['bbox', 'segm'], default='segm')
    parser.add_argument('--class-agnostic', action='store_true')
    parser.add_argument('--max-dets', type=int, default=1000)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.max_dets < 10:
        parser.error('--max-dets must be at least 10')
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    gt = COCO(args.annotations)
    with open(args.predictions, encoding='utf-8') as stream:
        predictions = json.load(stream)
    if not isinstance(predictions, list):
        raise ValueError('Expected a COCO result list with image_id/category_id/score.')
    # COCO derives segmentation areas from masks only when bbox is absent.
    if args.iou_type == 'segm':
        predictions = [{k: v for k, v in row.items() if k != 'bbox'} for row in predictions]
    if predictions:
        detections = gt.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = dict(images=gt.dataset['images'],
                                  categories=gt.dataset['categories'], annotations=[])
        detections.createIndex()
    evaluator = COCOeval(gt, detections, args.iou_type)
    evaluator.params.imgIds = sorted(gt.getImgIds())
    evaluator.params.maxDets = [1, 10, args.max_dets]
    evaluator.params.useCats = int(not args.class_agnostic)
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval['precision'][:, :, :, 0, -1]
    def mean_valid(values):
        values = values[values > -1]
        return float(values.mean()) if values.size else None
    metrics = dict(AP=mean_valid(precision),
                   AP50=mean_valid(precision[0]),
                   AP75=mean_valid(precision[5]),
                   max_dets=args.max_dets, iou_type=args.iou_type,
                   class_agnostic=args.class_agnostic)
    print(json.dumps(metrics, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(metrics, stream, indent=2)


if __name__ == '__main__':
    main()
