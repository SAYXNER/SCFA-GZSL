# SCFA-GZSL
-----Code------

Enhancing Generalized Zero-Shot Learning via Semantic Contrast and Feature Aggregation(SCFA-GZSL)
The source code of the proposed algorithm in this paper are given in this folder.

![image](Fig.1.png)
## Preparation
### 1、Requirements
The implementation runs on

Python 3.7

torch 1.12.0

Numpy

Sklearn

Scipy

### 2、Datasets
The extracted features for AWA datasets are from [[1]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning), FLO and CUB datasets are from [[2]](https://github.com/yunlongyu/EPGN). 

## Preparation Training & Test
Exemplar commands are listed here for AwA1 dataset

+ You can
```
python ../scripts/train.py --dataset AWA1 --ga 0.5 --nSample 5000 --gpu 1 --lr 0.00003 \ --classifier_lr 0.003 --kl_warmup 0.01  --vae_dec_drop 0.5 --vae_enc_drop 0.4 \ --gen_nepoch 420 --evl_start 200 --evl_interval 400  --manualSeed 6152
```
