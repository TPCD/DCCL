# DCCL

Offical implementation of our Dynamic Conceptional Contrastive Learning for Generalized Category Discovery in CVPR2023 ([arXiv](https://arxiv.org/pdf/2303.17393.pdf)) by Nan Pu, Zhun Zhong, Nicu Sebe.

## Abstract
Generalized category discovery (GCD) is a recently proposed open-world problem, which aims to automatically cluster partially labeled data. The main challenge is that the unlabeled data contain instances that are not only from known categories of the labeled data but also from novel categories. This leads traditional novel category discovery (NCD) methods to be incapacitated for GCD, due to their assumption of unlabeled data are only from novel categories. One effective way for GCD is applying selfsupervised learning to learn discriminate representation for unlabeled data. However, this manner largely ignores underlying relationships between instances of the same concepts (e.g., class, super-class, and sub-class), which results in inferior representation learning. In this paper, we propose a Dynamic Conceptional Contrastive Learning (DCCL) framework, which can effectively improve clustering accuracy by alternately estimating underlying visual conceptions and learning conceptional representation. In addition, we design a dynamic conception generation and update mechanism, which is able to ensure consistent conception learning and thus further facilitate the optimization of DCCL. Extensive experiments show that DCCL achieves new state-of-the-art performances on six generic and fine-grained visual recognition datasets, especially on fine-grained ones. For example, our method significantly surpasses the best competitor by 16.2% on the new classes for the CUB-200 dataset. 

![image](https://github.com/TPCD/DCCL/blob/main/DCCL_framework.png)

## Requirements
- Python 3.8
- Pytorch 1.10.0
- torchvision 0.11.1
```
pip install -r requirements.txt
```

## Datasets
In our experiments, we use generic image classification datasets including [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet](https://image-net.org/download.php).

We also use fine-grained image classification datasets including [CUB-200](https://www.kaggle.com/datasets/coolerextreme/cub-200-2011/versions/1), [Stanford-Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), and [Oxford-Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Pretrained Checkpoints
Our model is initialized with the parameters pretrained by DINO on ImageNet.
The DINO checkpoint of ViT-B-16 is available at [here](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth).

## Training and Evaluation Instructions
### Step 1. Set config
Set the path of datasets and the directory for saving outputs in ```config.py```.
### Step 2. Train and Test on CUB200 dataset
python G0_CUB200.py

## Experiments on Other datasets
For experiments on other datasets, please modify parser.add_argument('--dataset_name', type=str, default='cub', help='options: imagenet_100,cifar10, cifar100, scars') in the demo code and refer to the paper for other hyperparameters' setting.

## Results
Results of our method are reported as below. 
| **Datasets**       | **All** | **Old** | **New** | 
|:------------|:--------:|:---------:|:---------:|
| CIFAR10 | 96.3| 96.5| 96.9 |
| CIFAR100 | 75.3 |76.8 |70.2 | 
| ImageNet-100 |   80.5| 90.5| 76.2 | 
| CUB-200 | 63.5 | 60.8 | 64.9  | 
| Stanford-Cars | 43.1|  55.7 | 36.2 |
| Oxford-Pet | 88.1|  88.2 | 88.0  | 

## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@inproceedings{pu2023dynamic,
  title={Dynamic Conceptional Contrastive Learning for Generalized Category Discovery},
  author={Pu, Nan and Zhong, Zhun and Sebe, Nicu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7579--7588},
  year={2023}
}
```

## Acknowledgement
This project is modified from https://github.com/YiXXin/XCon. Thanks for their nice work.