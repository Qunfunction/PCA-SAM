"""Check metric behavior on tiny synthetic COCO inputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def evaluate(self, predictions, class_agnostic=False):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            truth = dict(info={}, images=[dict(id=1, width=32, height=32)],
                         categories=[dict(id=1, name='normal_tumor'),
                                     dict(id=2, name='immune')],
                         annotations=[dict(id=1, image_id=1, category_id=1,
                                           bbox=[4, 4, 12, 12], area=144, iscrowd=0)])
            (tmp / 'gt.json').write_text(json.dumps(truth))
            (tmp / 'pred.json').write_text(json.dumps(predictions))
            command = [sys.executable, str(ROOT / 'tools/evaluate_coco.py'),
                       str(tmp / 'gt.json'), str(tmp / 'pred.json'),
                       '--iou-type', 'bbox', '--output', str(tmp / 'metrics.json')]
            if class_agnostic:
                command.append('--class-agnostic')
            subprocess.run(command, check=True, capture_output=True, text=True)
            return json.loads((tmp / 'metrics.json').read_text())

    def test_perfect_detection(self):
        result = self.evaluate([dict(image_id=1, category_id=1,
                                     bbox=[4, 4, 12, 12], score=.9)])
        self.assertAlmostEqual(result['AP'], 1.)

    def test_category_policy(self):
        predictions = [dict(image_id=1, category_id=2, bbox=[4, 4, 12, 12], score=.9)]
        self.assertAlmostEqual(self.evaluate(predictions)['AP'], 0.)
        self.assertAlmostEqual(self.evaluate(predictions, class_agnostic=True)['AP'], 1.)

    def test_empty_predictions(self):
        self.assertAlmostEqual(self.evaluate([])['AP'], 0.)


if __name__ == '__main__':
    unittest.main()
