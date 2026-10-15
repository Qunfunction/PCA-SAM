# Source Release Validation

Validation used official MMDetection 3.2.0, isolated from the original experiment
checkout, with Python 3.8, PyTorch 1.13.1+cu116, MMCV 2.1.0 and MMEngine 0.10.7.

## Checkpoint compatibility

The existing PCA-SAM seed-20261012 epoch-12 checkpoint loaded strictly into both
the experiment implementation and the packaged implementation: 550 state-dict
entries, with no missing or unexpected keys.

On one 500 x 500 HHG image, each implementation returned 171 instances. Bounding
boxes, scores, labels and all binary mask pixels were exactly equal:

| Output | Shape | Maximum absolute difference |
| --- | --- | --- |
| Bounding boxes | 171 x 4 | 0 |
| Scores | 171 | 0 |
| Labels | 171 | 0 |
| Masks | 171 x 500 x 500 | 0 |

This is a regression check for packaging and renaming, not a new cohort-level
performance evaluation or a rerun of all reported experiments. Images and
checkpoint weights used by the check are not part of this source distribution.

## Automated checks

Model tests cover registration, CSPC feature sizes and gradients, frozen SAM
components, contextual box outputs, student and final mask sizes, real SAM
teacher targets aligned to 28 x 28, cascade loss/backward plumbing with synthetic
teacher targets, and inference that must not call the teacher.

Evaluator tests use synthetic boxes to check a perfect match, wrong-class
behavior under both category policies, and an empty prediction list.
