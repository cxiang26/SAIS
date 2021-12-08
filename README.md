# SAIS: Single-stage Anchor-free Instance Segmentation

## Introduction

This is an official release of the paper **SAIS: Single-stage Anchor-free Instance Segmentation**. 
Global-based and Local-based instance encoding methods are implemented.
## Results

### Instance Segmentation on COCO （FPS including forward and whole post-processing）

| Method      | Backbone      | Lr Schd | Mask mAP| mAP@S | mAP@L | FPS   |
| :---:       | :---:         | :---:   | :---:   | :---: | :---: | :---: |
| SAIS-L-544  | ResNet-50-FPN |  2x     | 29.2    | 10.7  | 38.2  | 34.0  |
| SAIS-G-544  | ResNet-50-FPN |  2x     | 27.2    | 7.5   | 46.4  | 38.6  |
| SAIS-GL-544 | ResNet-50-FPN |  2x     | 28.6    | 7.8   | 50.3  | 38.6  |
| :---:       | :---:         | :---:   | :---:   | :---: | :---: | :---: |
| SAIS-L-704  | ResNet-101-FPN | 3x     | 33.7    | 14.0  | 52.1  | 26.4  |
| SAIS-G-704  | ResNet-101-FPN | 3x     | 32.8    | 11.1  | 52.6  | 28.6  |
| SAIS-GL-704 | ResNet-101-FPN | 3x     | 33.0    | 12.2  | 53.0  | 28.6  |
 
## Installation

It requires the following OpenMMLab packages:

- MIM = 0.1.5
- MMCV-full = v1.3.14
- MMDetection = v2.17.0
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