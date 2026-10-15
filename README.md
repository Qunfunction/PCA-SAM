# PCA-SAM

**Photonic-contrast-aware SAM for Cell Identification in Label-free Third Harmonic
Generation Images of Lung and Pleural Biopsies**

PCA-SAM extends label-free higher harmonic generation (HHG) imaging toward
automated cell-level analysis of human lung and pleural biopsies, with third
harmonic generation (THG) providing the dominant cellular contrast. It embeds
a frozen SAM encoder within a unified multitask framework for cell detection,
nuclear instance segmentation, and cell-class recognition, without manual
prompts or a separate external detector at inference.

## Cell-level analysis in HHG and THG imaging

THG reveals tissue structures without staining, but low-signal nuclei can
resemble tissue gaps. Dense spatial arrangements and heterogeneous cell
populations further complicate instance separation and recognition. Local
appearance therefore needs to be interpreted within the surrounding tissue.

PCA-SAM incorporates tissue context at both feature and instance levels,
connecting contextual target identification with localization, classification,
and contour learning through shared features and joint optimization of
downstream components.

- **Cross-scale photonic semantic calibration (CSPC)** combines shallow appearance
  features with spatially pooled deep context through asymmetric cross-attention,
  bringing surrounding tissue information into the representation of ambiguous
  negative-contrast targets.
- **Cascaded boundary reconstruction decoder (CBRD)** combines candidate-region
  features with surrounding tissue information for cascaded localization and
  classification, alongside SAM-guided auxiliary mask learning for contour
  prediction.
- **Unified, prompt-free inference** produces bounding boxes, nuclear instance
  masks, and class labels within one framework. Candidate regions are generated
  internally, and the SAM teacher is used only during training.

The resulting nuclear contours and instance locations provide a basis for
morphometry and cell-density mapping, linking label-free optical images to
quantitative descriptions of cell populations for biopsy analysis and targeted
histological review.

## Framework

```text
Lung and pleural biopsy HHG images (THG-dominant cellular contrast)
  -> Frozen SAM encoder
  -> CSPC: local appearance + tissue context
  -> Internal candidate-region generation
  -> CBRD: contextual instance prediction + contour learning
  -> Bounding boxes, nuclear instance masks, and cell-class labels
```

The released configuration recognizes `normal_tumor` (normal and tumor cells
combined) and `immune`. Instance masks delineate nuclei.

## LP-HHG

LP-HHG comprises nearly 1,300 images and over 40,000 annotated instances from
44 patients, constructed from public clinical HHG images. Evaluation uses a
patient-disjoint partition. Images, annotations, patient split files, and trained
PCA-SAM checkpoints are not included in this repository.

## Getting started

### Installation

The supported environment uses Python 3.8, PyTorch 1.13.1 with CUDA 11.6,
and MMDetection 3.2.0. Run the following from the repository root:

```bash
python -m pip install "torch==1.13.1+cu116" "torchvision==0.14.1+cu116" --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install openmim
mim install "mmcv==2.1.0"
python -m pip install -r requirements.txt
python -m pip install -e .
```

### SAM initialization

Download the [SAM ViT-B image encoder](https://github.com/Qunfunction/PCA-SAM/releases/download/v0.1.0/image_encoder_vit_b.pth)
and place it under `checkpoints/sam_vit_b/`.

Training also uses the original SAM prompt encoder and mask decoder for auxiliary
supervision. Prepare these locally from the
[official SAM ViT-B checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints):

```bash
python tools/prepare_sam_weights.py sam_vit_b_01ec64.pth --output-dir checkpoints/sam_vit_b
```

Use an empty output directory when preparing all three components. Trained
PCA-SAM weights are not distributed; evaluation and inference require a trained
PCA-SAM checkpoint, not the SAM encoder alone.

### Data and usage

Organize images and two-class COCO annotations as follows, keeping all images
and patches from a patient in the same subset:

```text
data/LP-HHG/
  images/
  annotations/
    instances_train.json
    instances_test.json
```

COCO category IDs are 1 (`normal_tumor`) and 2 (`immune`). Training, evaluation,
and prediction entry points are:

```bash
python tools/train.py configs/pca_sam_vit_b_lphhg.py --work-dir work_dirs/pca_sam
python tools/test.py configs/pca_sam_vit_b_lphhg.py path/to/pca_sam_checkpoint.pth
python tools/predict.py configs/pca_sam_vit_b_lphhg.py path/to/pca_sam_checkpoint.pth path/to/image.png --output outputs/prediction.json --visualization outputs/prediction.png
```

Model components are in `pca_sam/models/`, configurations in `configs/`, and
training and inference tools in `tools/`.

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
