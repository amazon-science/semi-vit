## Semi-ViT: Semi-supervised Vision Transformers at Scale


This is a PyTorch implementation of the paper [Semi-ViT](https://arxiv.org/abs/2208.05688). It is a state-of-the-art semi-supervised learning of vision transformers.

If you use the code/model/results of this repository please cite:
```
@inproceedings{cai2022semi,
  author  = {Zhaowei Cai and Avinash Ravichandran and Paolo Favaro and Manchen Wang and Davide Modolo and Rahul Bhotika and Zhuowen Tu and Stefano Soatto},
  title   = {Semi-supervised Vision Transformers at Scale},
  booktitle = {NeurIPS},
  Year  = {2022}
}
```

### Install

First, [install PyTorch](https://pytorch.org/get-started/locally/) and torchvision. We have tested on version of 1.7.1, but newer versions should also be working.

```bash
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
``` 

Also install other dependencies, e.g.,

```bash
$ pip install timm==0.4.5
``` 


### Data Preparation

Assume ImageNet folder is ``~/data/imagenet/``, install ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet), with the standard data folder structure for the torchvision ``datasets.ImageFolder``. Please download the ImageNet [index files](https://eman-cvpr.s3.amazonaws.com/imagenet_indexes.zip) for semi-supervised learning experiments. The file structure should look like:

  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   └── *.jpeg
  │   ├── class2
  │   │   └── *.jpeg
  │   └── ...
  ├── val
  │   ├── class1
  │   │   └── *.jpeg
  │   ├── class2
  │   │   └── *.jpeg
  │   └── ...
  └── indexes
      └── *_index.csv
  ```

Please also download the [MAE self-pretrained weights](https://github.com/facebookresearch/mae), and move them to the folder of ``pretrain_weights``.

### Supervised Finetuning

The supervised finetuning instruction is in [FINETUNE.md](FINETUNE.md).

### Semi-supervised Finetuning

The semi-supervised finetuning instruction is in [SEMIVIT.md](SEMIVIT.md).

### Results

If the model is self-pretrained, the results would be close to the following (with some minor variance):

| model | method | acc@1% IN | acc@10% IN | acc@100% IN |
| :---: | :---: | :---: | :---: | :---: |
| ViT-Base | Finetune | 57.4 | 73.7 | 83.7 |
| ViT-Base | Semi-ViT | 71.0 | 79.7 | - |
| ViT-Large | Finetune | 67.1 | 79.2 | 86.0 |
| ViT-Large | Semi-ViT | 77.3 | 83.3 | - |
| ViT-Huge | Finetune | 71.5 | 81.4 | 86.9 |
| ViT-Huge | Semi-ViT | 80.0 | 84.3 | - |

If the model is not self-pretrained, the results would be close to the following (with some minor variance):

| model | method | acc@10% IN |
| :---: | :---: | :---: |
| ViT-Small | Finetune | 56.2 |
| ViT-Small | Semi-ViT | 70.9 |
| ViT-Base | Finetune | 57.0 |
| ViT-Base | Semi-ViT | 73.5 |
| ConvNeXT-Tiny | Finetune | 61.2 |
| ConvNeXT-Tiny | Semi-ViT | 74.1 |
| ConvNeXT-Small | Finetune | 64.1 |
| ConvNeXT-Small | Semi-ViT | 75.1 |

### License

This project is under the Apache-2.0 license. See [LICENSE](LICENSE) for details.
