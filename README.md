# Code Execution Manual for AlignPro

## Paper Information

> **Improving Alignment for Few-Shot Class-Incremental Learning via Prompt-Tuned Vision Transformer** <br>Yilei Qian, Linfeng Xu, Xiao Hou, Kanglei Geng, Guanheng Huang, Shaoxu Cheng, Hongliang Li
> The paper has been submitted to *Pattern Recognition* and is currently under review.

## Dependencies and Installation

**Depedencies**

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Pytorch 2.0 or later. See [Pytorch]( https://pytorch.org) for install instructions.
- Linux (Ubuntu 18.04.3)

**Installation**

First, you can clone this repo using the command:

```shell 
git clone https://github.com/mon1ystar/AlianPro.git
```

Then, you can create a virtual environment using conda, as follows:

```shell
conda env create -f environment.yaml
conda activate tbw
```

## Data preparation

We provide source about the datasets we use in our experiment as below:

| Dataset       | Dataset                                                      |
| ------------- | ------------------------------------------------------------ |
| ARIC          | [ARIC](https://ivipclab.github.io/publication_ARIC/ARIC/)    |
| CIFAR100      | [CIFAR100](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) |
| CUB200        | [CUB200](https://www.vision.caltech.edu/datasets/cub_200_2011/) |
| mini-imagenet | [miniimagenet](https://huggingface.co/datasets/timm/mini-imagenet) |

## Training && Evaluation

Run the following command to train and test the model sequentially:

#### CIFAR100

```shell
bash ./train_cifar100.sh
```

#### CUB200

```shell
bash ./train_cub200.sh
```

#### mini-imagenet

```shell
bash ./train_miniimagenet.sh
```

#### ARIC

```shell
bash ./train_aric.sh
```

After finished, you can get model checkpoints and watch the log file in the folder `./checkpoint`

