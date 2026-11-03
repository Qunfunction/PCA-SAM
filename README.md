# PCA-SAM

**Photonic-contrast-aware SAM for Cell Identification in Label-free Third Harmonic
Generation Images of Lung and Pleural Biopsies**

Label-free HHG microscopy enables rapid imaging of fresh lung and pleural
biopsies, but extracting quantitative cellular information remains challenging.
PCA-SAM integrates SAM into a unified framework that combines local THG contrast
with cellular morphology and surrounding tissue context to automatically detect,
segment, and classify cells without external prompts. It supports quantitative
analysis of nuclear morphology and cellular distribution, advancing label-free
biopsy imaging toward automated and intelligent tissue assessment.

The manuscript is currently in the submission process, and the LP-HHG dataset
is being prepared for public release. Updates on dataset availability will be
posted in this repository.

## 中文介绍

无标记HHG显微成像能够快速获取新鲜肺及胸膜活检组织图像，但定量细胞信息的提取仍面临挑战。PCA-SAM将SAM整合至统一框架，融合局部THG对比、细胞形态与周围组织信息，无需外部提示即可自动完成细胞检测、实例分割和分类。该方法支持细胞核形态及细胞空间分布的定量分析，推动无标记活检成像向自动化、智能化组织评估发展。

目前论文正处于投稿阶段，LP-HHG数据集正在整理，计划后续开源。数据集开放的相关进展将在本仓库更新。

## LP-HHG dataset

LP-HHG contains approximately 1,300 HHG images from more than 40 patients,
covering normal/benign tissue, pre-invasive lesions, and malignant tumors.
Training and test patients are disjoint. The three cell classes are tumor
(blue), immune (red), and normal (white). Normal cells refer to non-immune cells
in normal tissue; non-immune cells in other pathological tissues are
operationally grouped as tumor cells.

## Public datasets

- **TissueNet (fluorescence):** [Official download](https://datasets.deepcell.org/data).
- **LIVECell (phase contrast):** [Official image and annotation downloads](https://sartorius-research.github.io/LIVECell/#how-to-access-livecell).

## Quick start

### 1. Installation

Use Python 3.8 with CUDA 11.6:

```bash
git clone https://github.com/Qunfunction/PCA-SAM.git
cd PCA-SAM
python -m pip install "torch==1.13.1+cu116" "torchvision==0.14.1+cu116" --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install openmim
mim install "mmcv==2.1.0"
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. Weights and data

Download the [official SAM ViT-B checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints)
and extract its components into an empty directory:

```bash
python tools/prepare_sam_weights.py sam_vit_b_01ec64.pth --output-dir checkpoints/sam_vit_b
```

Place images in `data/LP-HHG/images/` and COCO annotations in
`data/LP-HHG/annotations/instances_train.json` and `instances_test.json`.
Use categories `1: tumor`, `2: immune`, and `3: normal`, in that order.

### 3. Training

```bash
python tools/train.py configs/pca_sam_vit_b_lphhg_three_class.py --work-dir work_dirs/pca_sam_three_class
```

The configuration trains for 12 epochs. For the earlier two-class setup, use
`configs/pca_sam_vit_b_lphhg.py` with matching annotations and weights.

### 4. Evaluation and prediction

Replace the checkpoint and image paths with your own:

```bash
python tools/test.py configs/pca_sam_vit_b_lphhg_three_class.py path/to/checkpoint.pth
python tools/predict.py configs/pca_sam_vit_b_lphhg_three_class.py path/to/checkpoint.pth path/to/image.png --output outputs/prediction.json --visualization outputs/prediction.png
```

Evaluation and prediction require a trained PCA-SAM checkpoint; trained weights
and LP-HHG data are not included in this repository.

## Citation

Please cite our manuscript when using this code:

```bibtex
@unpublished{sun_pca_sam,
  author = {Sun, Zequn and Wu, Yuchen and Bo, Qiyu and Wang, Xianghui and Zhang, Zhiqing and Chang, Shengjiang},
  title = {{PCA-SAM}: Photonic-contrast-aware {SAM} for Cell Identification in Label-free Third Harmonic Generation Images of Lung and Pleural Biopsies},
  note = {Manuscript}
}
```
