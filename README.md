# PCA-SAM

**Photonic-contrast-aware SAM for Cell Identification in Label-free Third Harmonic
Generation Images of Lung and Pleural Biopsies**

PCA-SAM embeds a frozen SAM image encoder in a unified multitask model for cell
detection, nuclear instance segmentation, and two-class recognition in HHG biopsy
images. An internal RPN generates candidate regions. Inference needs no manual
point/box prompts or separate external detector.

## Citation

Please cite the following manuscript when using PCA-SAM code in your research:

Zequn Sun, Yuchen Wu, Qiyu Bo, Xianghui Wang, Zhiqing Zhang, and Shengjiang Chang.
**PCA-SAM: Photonic-contrast-aware SAM for Cell Identification in Label-free Third
Harmonic Generation Images of Lung and Pleural Biopsies.**

```bibtex
@unpublished{sun_pca_sam,
  author = {Sun, Zequn and Wu, Yuchen and Bo, Qiyu and Wang, Xianghui and Zhang, Zhiqing and Chang, Shengjiang},
  title = {{PCA-SAM}: Photonic-contrast-aware {SAM} for Cell Identification in Label-free Third Harmonic Generation Images of Lung and Pleural Biopsies},
  note = {Manuscript}
}
```

## Architecture

```text
HHG image
  -> PCASAMImageEncoder (frozen SAM ViT-B)
  -> CSPCNeck (feature alignment, pooled-context cross-attention, PAFPN)
  -> internal RPN
  -> CBRDHead
       |-- contextual cascaded localization and classification
       |-- auxiliary student mask features + final mask prediction
  -> boxes, nuclear instance masks, two-class labels

Training only:
  internal positive boxes + center points
  -> frozen SAM prompt encoder and mask decoder
  -> auxiliary mask targets for the student branch
```

- **CSPC**: cross-scale photonic semantic calibration. Encoder blocks 3, 6, 9,
  and 12 supply four 768-channel features at the same spatial resolution. Each
  depth is aligned to 256, 128, 64, and 32 pixel grids. Shallow features form
  128-channel queries; deep features form 128-channel keys and values, pooled
  to 32 x 32 when needed. One four-head attention module is shared across grids.
  Attention output is concatenated with unpooled deep features before PAFPN
  builds five 256-channel feature levels at strides 4, 8, 16, 32, and 64.
- **CBRD**: cascaded boundary reconstruction decoder. Object and enlarged
  contextual RoIs support localization and classification at three stages.
  Their feature representations are concatenated. A convolutional student
  learns SAM-derived auxiliary targets; its intermediate features are
  concatenated with original RoI features for the final nuclear mask head.
  The SAM teacher is not called during inference.

The output classes are `normal_tumor` (normal and tumor cells combined) and
`immune`. Masks follow nuclear contours. These are not separate normal-versus-
tumor diagnostic predictions or whole-cell HHG masks.

## Repository layout

```text
pca_sam/models/
  detector.py                 PCASAM unified detector
  necks/cspc.py               CSPCNeck
  roi_heads/cbrd.py           CBRDHead
  roi_heads/context_bbox_head.py
  roi_heads/auxiliary_mask_head.py
  roi_heads/boundary_mask_head.py
  sam/                       SAM encoder and training-teacher components
  rpn_head.py                mixed-precision-safe internal RPN
configs/
  _base_/pca_sam_vit_b.py     model architecture
  pca_sam_vit_b_lphhg.py      LP-HHG training and evaluation
tools/
  prepare_sam_weights.py     prepare official SAM initialization
  train.py                   train a selected seed
  test.py                    class-aware COCO evaluation
  predict.py                 single-image inference, JSON/RLE and overlay
  evaluate_coco.py           class-aware or class-agnostic COCO AP
tests/
  test_models.py             module, training-loss, and inference checks
docs/
  source_manifest.json       hashes for the pre-packaging model source
```

## Installation

Use an isolated environment. The local experiment runtime uses Python 3.8,
PyTorch 1.13.1+cu116, torchvision 0.14.1, MMCV 2.1.0, and MMEngine 0.10.7.
This package targets official MMDetection 3.2.0. Full MMCV with compiled ops is
required; `mmcv-lite` is insufficient.

For the CUDA 11.6 / PyTorch 1.13 environment:

```bash
python -m pip install "torch==1.13.1+cu116" "torchvision==0.14.1+cu116" --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install openmim
mim install "mmcv==2.1.0"
python -m pip install -r requirements.txt
python -m pip install -e .
```

For another CUDA/PyTorch combination, install a matching torchvision and MMCV
build using the [MMDetection installation guide](https://mmdetection.readthedocs.io/en/v3.2.0/get_started.html).
Run from the repository root. There is no dependency on a local experimental
checkout or machine-specific drive path.

## SAM initialization

The original SAM ViT-B image-encoder initialization is available as a
[release asset](https://github.com/Qunfunction/PCA-SAM/releases/download/v0.1.0/image_encoder_vit_b.pth),
named `image_encoder_vit_b.pth` (358,740,511 bytes). Trained PCA-SAM checkpoints
are not distributed. Keep the encoder under `checkpoints/sam_vit_b/` after
download. Its SHA-256 checksum is
`1e4e09033213cc0e89c25b2fef9d48739e325d1d25fff9961802dfd2edc4fcc7`.

Obtain `sam_vit_b_01ec64.pth` from the
[official Segment Anything repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).
Then split the official checkpoint into the initialization components:

```bash
python tools/prepare_sam_weights.py sam_vit_b_01ec64.pth --output-dir checkpoints/sam_vit_b
```

The output files are `image_encoder_vit_b.pth`, `mask_decoder_vit_b.pth`, and
`prompt_encoder_vit_b.pth`. The released training configuration also requires
the original prompt-encoder and mask-decoder weights for its frozen SAM teacher;
the image encoder alone is not sufficient to initialize this training setup.
These two teacher files can be prepared locally with the command above and are
not included in the release assets. A full trained PCA-SAM checkpoint contains these
components already, so the testing and prediction tools do not require the
separate initialization files.

## LP-HHG data

LP-HHG is constructed from public clinical HHG images with instance annotations
and a patient-disjoint partition for PCA-SAM training and evaluation.
Use the instance annotations and patient partition associated with the experiment
being reproduced, organized in COCO format:

```text
data/LP-HHG/
  images/
  annotations/
    instances_train.json
    instances_test.json
```

Category IDs are 1 (`normal_tumor`) and 2 (`immune`); internal model labels are
0 and 1. Each instance needs a bounding box, segmentation, area, and category.
Keep all source images and derived patches from one patient in one subset.
Set `PCA_SAM_DATA_ROOT` to use another dataset location and `PCA_SAM_SAM_ROOT`
to use another directory of SAM initialization files.

This source release contains model code and configuration. LP-HHG images,
annotations, patient split files, and trained PCA-SAM weights are not bundled
or published by this commit. The original public image resource does not
replace the LP-HHG instance annotations or its fixed split.

## Training

```bash
python tools/train.py configs/pca_sam_vit_b_lphhg.py --seed 20261011 --work-dir work_dirs/pca_sam_seed_20261011
python tools/train.py configs/pca_sam_vit_b_lphhg.py --seed 20261012 --work-dir work_dirs/pca_sam_seed_20261012
python tools/train.py configs/pca_sam_vit_b_lphhg.py --seed 20261013 --work-dir work_dirs/pca_sam_seed_20261013
```

The released configuration uses 12 epochs, batch size 1 with two-step gradient
accumulation, AMP, SGD (learning rate 0.0025, momentum 0.9, weight decay 0.0001),
500 warmup iterations, and learning-rate reductions at epochs 8 and 11. The
final epoch checkpoint is used; validation and test-driven checkpoint selection
are disabled. Cascade positive IoU thresholds are 0.5, 0.6, and 0.7, and stage
loss weights are 1, 0.5, and 0.25. Inputs are resized to 1024 x 1024 and converted
from BGR to RGB before SAM normalization.

The internal RPN uses anchor scales 4 and 8, 6,000 pre-NMS candidates per
level, and up to 3,000 proposals. Final predictions use Soft-NMS with IoU 0.5,
a score threshold of 0.05, and a cap of 1,000 instances per image.

Resume with the same configuration and output directory:

```bash
python tools/train.py configs/pca_sam_vit_b_lphhg.py --seed 20261011 --work-dir work_dirs/pca_sam_seed_20261011 --resume
```

## Evaluation

```bash
python tools/test.py configs/pca_sam_vit_b_lphhg.py work_dirs/pca_sam_seed_20261011/epoch_12.pth
```

The configured evaluator reports bounding-box and mask AP over IoU 0.50:0.05:0.95,
including classwise results, with up to 1,000 detections per image. Compare
methods with the same annotations, split, area range, detection cap, and
evaluation implementation. A different detection cap can change AP.

For COCO-format prediction JSON files, the standalone evaluator supports both
class-aware and class-agnostic metrics:

```bash
python tools/evaluate_coco.py data/LP-HHG/annotations/instances_test.json outputs/mask_predictions.json --iou-type segm --class-agnostic --max-dets 1000
```

This repository does not report newly reproduced manuscript scores or confidence
intervals. Statistical tables require the original run-level predictions and
patient mapping; no intervals are synthesized by the evaluation tools.

## Inference

```bash
python tools/predict.py configs/pca_sam_vit_b_lphhg.py checkpoints/pca_sam_epoch12.pth path/to/image.png --output outputs/prediction.json --visualization outputs/prediction.png
```

The JSON contains `bbox_xyxy`, model label, category ID, class name, score, and
COCO RLE segmentation for each retained instance. `--score-thr` defaults to 0.3
for export and visualization; it does not change the published test protocol.
Existing output files are not overwritten. Use `--device cpu` if CUDA is
unavailable, with substantially slower execution expected.

## Checkpoint compatibility

Project-facing class and module names follow PCA-SAM, CSPC, and CBRD. Internal
parameter paths such as `backbone`, `neck.fpn`, `neck.cross_attention`,
`roi_head.bbox_head`, and `roi_head.sam_mask_head` are retained for compatibility
with the experiment checkpoints. No parameter-key conversion is required.
The SAM encoder and teacher components remain frozen even when loading their
parameters from a full model checkpoint.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover configuration registration, CSPC shapes and gradients, contextual
box prediction, student/final mask branches, teacher-target alignment, training
losses, teacher-free inference, and COCO category policies. A strict checkpoint
load and single-image prediction-equivalence check are recorded in
[docs/VALIDATION.md](docs/VALIDATION.md).
