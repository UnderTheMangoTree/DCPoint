# DCPoint: Global-Local Dual Contrast for Self-Suoervised Representation Learning of 3D Point Clouds

The code was tested with the following environment: Ubuntu 18.04, python 3.7, pytorch 1.8.1, and CUDA 11.1.

## Backbone
+ [x] DGCNN
+ [x] CurveNet

### Downstream Tasks

+ [ ] Shape Classification
+ [ ] Part Segmentation
+ [ ] Semantic Segmentation

## DataSets

Please download the used dataset from the public resources of [CrossPoint](https://github.com/MohamedAfham/CrossPoint) [Download data](https://drive.google.com/drive/folders/1dAH9R3XDV0z69Bz6lBaftmJJyuckbPmR)

## Pre-training
Please run the following command:
```
python DCPoint\train.py
```
You need to edit the config file `DCPoint/config/config.yaml` to switch different backbone architectures (currently including `dgcnn-cls, curvenet-cls`).

## Acknowledgement
We would like to thank the [CrossPoint](https://github.com/MohamedAfham/CrossPoint) and [STRL](https://github.com/yichen928/STRL.git) for their open-source projects.



