# SAIS: Single-stage Anchor-free Instance Segmentation

## Introduction

This is an official release of the paper **SAIS: Single-stage Anchor-free Instance Segmentation**. 
Global-based and Local-based instance encoding methods are implemented.
## Results

### Instance Segmentation on COCO

| Backbone | Method | Lr Schd | Mask mAP| Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: |

## Installation

It requires the following OpenMMLab packages:

- MIM >= 0.1.5
- MMCV-full >= v1.3.14
- MMDetection >= v2.17.0
- scipy

```bash
pip install openmim scipy mmdet
mim install mmcv-full
```

## Usage

### Data preparation

Prepare data following [MMDetection](https://github.com/open-mmlab/mmdetection). The data structure looks like below:

```text
data/
├── coco
│   ├── annotations
│   │   ├── instance_{train,val}2017.json
│   ├── train2017
│   ├── val2017
│   ├── test2017

```

### Training and testing

For training and testing, you can directly use mim to train and test the model

## Citation

```bibtex
@inproceedings{xiang,
    title={{SAIS:} Single-stage Anchor-free Instance Segmentation},
    author={},
    year={},
    booktitle={},
}
```