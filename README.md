# PCA-SAM

**Photonic-contrast-aware SAM for Cell Identification in Label-free Third Harmonic
Generation Images of Lung and Pleural Biopsies**

PCA-SAM connects rapid, label-free higher harmonic generation (HHG) imaging with
automated cell-level analysis of human lung and pleural biopsies. With third
harmonic generation (THG) providing the dominant cellular contrast, the framework
brings cell localization, nuclear contour delineation, and cell-class recognition
into a unified analysis of heterogeneous tissue.

## Interpreting cells within their tissue context

THG makes tissue structures visible without staining, yet its optical contrast
does not always make individual cells easy to identify. Low-signal nuclei can
resemble tissue gaps, while densely arranged cells and heterogeneous populations
complicate both instance separation and recognition. Interpreting these regions
requires information beyond their local appearance.

PCA-SAM brings surrounding tissue context into both feature representations and
instance prediction, linking contextual target identification with contour
learning. A frozen SAM encoder is embedded within a unified multitask framework,
with downstream components jointly optimized for detection, segmentation, and
classification. Inference requires neither user-provided prompts nor a separate
external detector, allowing cell-level predictions to be obtained directly from
HHG biopsy images.

## From optical images to quantitative tissue information

Cell locations, nuclear contours, and class labels turn label-free images into
measurable morphological and spatial information. These outputs provide a basis
for nuclear morphometry and cell-density mapping, supporting quantitative
characterization of cell populations and targeted histological review.

By linking the optical characteristics of THG with tissue-context interpretation
and joint instance analysis, PCA-SAM provides a route from rapid HHG image
acquisition to automated descriptions of cellular organization in biopsy tissue.

## LP-HHG

The study introduces LP-HHG, constructed from public clinical HHG images, with
nearly 1,300 images and over 40,000 annotated instances from 44 patients.
Patient-disjoint evaluation assesses cell-level analysis on held-out patients.

## Availability

This repository provides the PCA-SAM implementation and model configurations.
The [original SAM ViT-B image encoder](https://github.com/Qunfunction/PCA-SAM/releases/tag/v0.1.0)
is available separately. Trained PCA-SAM checkpoints, LP-HHG images, annotations,
and patient split files are not included in this release.

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
