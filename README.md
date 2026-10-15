# PCA-SAM

**Photonic-contrast-aware SAM for Cell Identification in Label-free Third Harmonic
Generation Images of Lung and Pleural Biopsies**

Rapid, label-free higher harmonic generation (HHG) imaging reveals biopsy tissue
structure, yet extracting quantitative cell-level information remains challenging
in third harmonic generation (THG)-dominant images. Low-signal nuclei can resemble
tissue gaps, while dense cellular arrangements and heterogeneous populations
complicate instance separation and recognition.

PCA-SAM interprets these ambiguous targets within their surrounding tissue,
embedding a frozen SAM encoder in a unified multitask framework that connects
cell localization, nuclear contour delineation, and cell-class recognition.
Tissue context is incorporated at both feature and instance levels, with joint
optimization of downstream components and no user-provided prompts or separate
external detector at inference. The resulting contours, locations, and class
labels provide a basis for nuclear morphometry and cell-density mapping,
linking rapid HHG imaging to automated, quantitative characterization of cell
populations in human lung and pleural biopsies.

The manuscript is currently in the submission process, and the LP-HHG dataset
is being prepared for public release. Updates on dataset availability will be
posted in this repository.

## 中文介绍

快速、无标记的高次谐波产生（HHG）成像能够揭示活检组织结构，但从以三次谐波产生（THG）为主导对比度的图像中提取定量细胞级信息仍面临挑战。低信号细胞核与组织空隙可能具有相似外观，而密集的细胞排列和异质细胞群体进一步增加了实例分离与识别的难度。

PCA-SAM结合周围组织信息辨识这些模糊目标，将冻结的SAM编码器内嵌于统一多任务框架，连接细胞定位、核轮廓勾画与类别识别。框架在特征和实例两个层面引入组织上下文，并联合优化下游预测组件，推理无需用户提示或独立的外部检测器。输出的轮廓、位置与类别标签为细胞核形态测量和细胞密度映射提供基础，将快速HHG成像连接到肺部和胸膜活检中细胞群体的自动化定量表征。

目前论文正处于投稿阶段，LP-HHG数据集正在整理，计划后续开源。数据集开放的相关进展将在本仓库更新。

## Workflow

### 1. Get the code

```bash
git clone https://github.com/Qunfunction/PCA-SAM.git
cd PCA-SAM
```

### 2. Set up the environment

Use Python 3.8 and a CUDA 11.6-compatible environment:

```bash
python -m pip install "torch==1.13.1+cu116" "torchvision==0.14.1+cu116" --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install openmim
mim install "mmcv==2.1.0"
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 3. Prepare SAM weights

Download the [SAM ViT-B image encoder](https://github.com/Qunfunction/PCA-SAM/releases/download/v0.1.0/image_encoder_vit_b.pth)
to `checkpoints/sam_vit_b/`.

Training also needs the original SAM prompt encoder and mask decoder. To prepare
all three components, download the
[official SAM ViT-B checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints)
and run the following with an empty output directory:

```bash
python tools/prepare_sam_weights.py sam_vit_b_01ec64.pth --output-dir checkpoints/sam_vit_b
```

### 4. Prepare the data

Place images in `data/LP-HHG/images/` and COCO annotations in
`data/LP-HHG/annotations/instances_train.json` and `instances_test.json`.
Keep all images and patches from a patient in the same subset. Use category IDs
1 for normal/tumor cells combined and 2 for immune cells.

### 5. Train

```bash
python tools/train.py configs/pca_sam_vit_b_lphhg.py --work-dir work_dirs/pca_sam
```

### 6. Evaluate and predict

Replace the checkpoint and image paths with your own files:

```bash
python tools/test.py configs/pca_sam_vit_b_lphhg.py path/to/pca_sam_checkpoint.pth
python tools/predict.py configs/pca_sam_vit_b_lphhg.py path/to/pca_sam_checkpoint.pth path/to/image.png --output outputs/prediction.json --visualization outputs/prediction.png
```

Trained PCA-SAM checkpoints and LP-HHG data are not included. Evaluation and
prediction require a trained PCA-SAM checkpoint, not the SAM encoder alone.

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
