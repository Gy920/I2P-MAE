# Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-3d-representations-from-2d-pre/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=learning-3d-representations-from-2d-pre)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-3d-representations-from-2d-pre/3d-point-cloud-linear-classification-on)](https://paperswithcode.com/sota/3d-point-cloud-linear-classification-on?p=learning-3d-representations-from-2d-pre)

Official implementation of ['Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders'](https://arxiv.org/pdf/2212.06785.pdf).

The paper has been accepted by **CVPR 2023** 🔥.

## News
* 📣 Please check our latest work [Point-NN, Parameter is Not All You Need](https://github.com/ZrrSkywalker/Point-NN) accepted by **CVPR 2023**, which, for the first time, acheives 3D understanding with $\color{darkorange}{No\ Parameter\ or\ Training\.}$ 💥
* 📣 Please check our latest work [PiMAE](https://github.com/BLVLab/PiMAE) accepted by **CVPR 2023**, which promotes 3D and 2D interaction to improve 3D object detection performance.
* The 3D-only variant of I2P-MAE is our previous work, [Point-M2AE](https://arxiv.org/pdf/2205.14401.pdf), accepted by **NeurIPS 2022** and [open-sourced](https://github.com/ZrrSkywalker/Point-M2AE).

## Introduction

Comparison with existing MAE-based 3D models on the three spilts of ScanObjectNN:
| Method | Parameters | GFlops| Extra Data | OBJ-BG | OBJ-ONLY| PB-T50-RS|
| :-----: | :-----: |:-----:| :-----: | :-----:| :-----:| :-----:|
| [Point-BERT](https://github.com/lulutang0608/Point-BERT) | 22.1M |4.8| -|87.43% |88.12% |83.07 %| 
| [ACT](https://github.com/RunpeiDong/ACT) | 22.1M |4.8| 2D|92.48%| 91.57% | 87.88% | 
| [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) | 22.1M |4.8| -|90.02%|88.29%|85.18%|
| [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE)| 15.3M |3.6| -|91.22%|88.81%|86.43%|
| **I2P-MAE** | **15.3M** |**3.6**| **2D**|**94.15%**|**91.57%**|**90.11%**|

We propose an alternative to obtain superior 3D representations from 2D pre-trained models via **I**mage-to-**P**oint Masked Autoencoders, named as **I2P-MAE**. By self-supervised pre-training, we leverage the well learned 2D knowledge to guide 3D masked autoencoding, which reconstructs the masked point tokens with an encoder-decoder architecture. Specifically, we conduct two types of image-to-point learning schemes: 2D-guided masking and 2D-semantic reconstruction. In this way, the 3D network can effectively inherit high-level 2D semantics learned from rich image data for discriminative 3D modeling.

<div align="center">
  <img src="pipeline.png"/>
</div>

## Requirements

### Installation
Create a conda environment and install basic dependencies:
```bash
git clone https://github.com/ZrrSkywalker/I2P-MAE.git
cd I2P-MAE

conda create -n i2pmae python=3.7
conda activate i2pmae

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```
Install GPU-related packages:
```bash
# Chamfer Distance and EMD
cd ./extensions/chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
### Datasets
For pre-training and fine-tuning, please follow [DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) to install ShapeNet, ModelNet40, ScanObjectNN, and ShapeNetPart datasets, referring to Point-BERT. Specially for Linear SVM evaluation, download the official [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) dataset and put the unzip folder under `data/`.

The final directory structure should be:
```
│I2P-MAE/
├──cfgs/
├──datasets/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──modelnet40_ply_hdf5_2048/  # Specially for Linear SVM
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──...
```


## Acknowledgement
This repo benefits from [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), and [CLIP](https://github.com/openai/CLIP). Thanks for their wonderful works.

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
