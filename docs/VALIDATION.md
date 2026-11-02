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

Model tests cover registration, MOMA feature sizes and gradients, frozen SAM
components, contextual box outputs, student and final mask sizes, real SAM
teacher targets aligned to 28 x 28, cascade loss/backward plumbing with synthetic
teacher targets, and inference that must not call the teacher.

Evaluator tests use synthetic boxes to check a perfect match, wrong-class
behavior under both category policies, and an empty prediction list.

## Manuscript naming update

The current manuscript calls the feature-fusion neck MOMA and the identification
head MTIH. Canonical registry names are `MOMANeck`, `MTIHHead`,
`MTIHContextBBoxHead`, `MTIHMaskHead`, and `MTIHAuxiliaryMaskHead`.
Earlier CSPC/CBRD imports and registry names remain aliases of the same classes.
The rename preserves module attributes and state-dict keys. The strict-load and
single-image results above describe the earlier packaging validation; they were
not rerun as part of this naming update.

`configs/pca_sam_vit_b_lphhg_three_class.py` adds the manuscript class scheme.
The existing `pca_sam_vit_b_lphhg.py` remains the legacy two-class configuration.
Changing the class count changes classification and mask output dimensions;
legacy two-class trained heads are not three-class trained heads. Neither the
three-class cohort metrics nor trained weights are supplied by this update.
The source hashes in `source_manifest.json` record historical experiment sources,
not hashes of the renamed current implementation files.
