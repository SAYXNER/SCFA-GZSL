# SCFA-GZSL
-----Code------

Enhancing Generalized Zero-Shot Learning through Semantic Contrast and Feature Aggregation(SCFA-GZSL)

The source code of the proposed algorithm in this paper are given in this folder.

![image](Fig.1.png)
## Preparation
### 1、Requirements
The implementation runs on

* Python 3.7

* torch 1.12.0

* h5py 3.8.0

* Numpy 1.21.5

* pillow 9.4.0

* scikit-learn 1.0.2

* Scipy 1.7.3

### 2、Datasets
The extracted features for AWA datasets are from [[1](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning)](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning), FLO and CUB datasets are from [[2](https://github.com/yunlongyu/EPGN)](https://github.com/yunlongyu/EPGN). 

## Preparation Training & Test
Exemplar commands are listed here for AwA1 dataset

+ You can
```
python ../scripts/train.py --dataset AWA1 --ga 0.5 --nSample 5000 --gpu 1 --lr 0.00003 \ --classifier_lr 0.003 --kl_warmup 0.01  --vae_dec_drop 0.5 --vae_enc_drop 0.4 \ --gen_nepoch 420 --evl_start 200 --evl_interval 400  --manualSeed 6152
```
Exemplar commands are listed here for AwA2 dataset

+ You can
```
python ../scripts/train.py --dataset AWA2 --ga 0.5 --nSample 5000 --gpu 1 --lr 0.00003 \ --classifier_lr 0.003 --kl_warmup 0.01  --vae_dec_drop 0.5 --vae_enc_drop 0.4 \
  --gen_nepoch 420 --evl_start 200 --evl_interval 400  --manualSeed 6152
```

Exemplar commands are listed here for CUB dataset

+ You can
```
python ../scripts/train.py --dataset CUB_STC  --ga 3 --nSample 1200 --gpu 0 \--lr 0.0001  --classifier_lr 0.005 --gen_nepoch 550 --kl_warmup 0.02 --weight_decay 1e-6 \
  --vae_enc_drop 0.4 --vae_dec_drop 0.5 --ae_drop 0.2 --cls_weight 0.01
```

Exemplar commands are listed here for FLO dataset

+ You can
```
python ../scripts/train.py --dataset FLO_EPGN --ga 5 --nSample 1500 --gpu 0 \--lr 0.0001  --classifier_lr 0.005 --kl_warmup 0.01 --gen_nepoch 800 \
  --vae_dec_drop 0.4 --vae_enc_drop 0.4 --ae_drop 0.2
```
## Supplementary Experimental Results

### GZSL

| Models       | AWA1 (U/S/H) | AWA2 (U/S/H) | CUB (U/S/H) | FLO (U/S/H) |
|--------------|--------------|--------------|-------------|-------------|
| LFGAA   | -/-/-        | 27.0/93.4/41.9| 36.2/80.9/50.0       | -/-/-    |
| DVBE    | -/-/-        | 63.6/70.8/41.9| 53.2/60.2/56.5| -/-/- |
| f-CLSWGAN| 57.9/61.4/59.6| -/-/-        |  43.7/57.7/49.7| 59.0/73.8/65.6 |
| f-VAEGAN-D2| -/-/-       |  57.6/70.6/63.5| 48.4/60.1 /53.6 |56.8/ 74.9/ 64.6|
| TF-VAEGAN | -/-/-        |  59.8 /75.1/ 66.6| 52.8/ 64.7 /58.1| 62.5 /84.1 /71.7|
| E-PGN    |  62.1 /83.4/ 71.2| 52.6/ 83.5/ 64.6 |52.0 /61.1/ 56.2| 71.5 /82.2 /76.5 |
| CADA-VAE | 57.3 /72.8/ 64.1| 55.8 /75.0 /63.9| 51.6/ 53.5/ 52.4| -/-/- |
| FREE     |  62.9/ 69.4/ 66.0| 60.4 /75.4 /67.1 |55.7/ 59.9 /57.7 |67.4/ 84.5/ 75.0|
| CE-GZSL  | 65.3 /73.4 /69.1|63.1 /78.6/ 70.0 |63.9 /66.8 /65.3 |69.0/ 78.7/ 73.5|
| SE-GZSL | 61.3 /76.7/ 68.1| 59.9/ 80.7/ 68.8| 53.1/ 60.3/ 56.4  | -/-/- | 
| AGZSL    | -/-/- |  65.1/ 78.9/ 71.3| 41.4 /49.7 /45.2|-/-/-|
| IZF      | 61.3/ 80.5 /69.1 |60.6/ 77.5/ 68.0| -/-/-  | 52.7/ 68.0/ 59.4| 
| SDGZSL   | -/-/-        | 64.6 /73.6/ 68.8| 59.9/ 66.4 /63.0 |83.3/ 90.2 /86.6 | 
| Ours    | 67.1 /80.8/ 73.3 |67.9/ 81.2/ 73.9 |63.8 /65.7/ 64.7 |86.8 /91.5 /89.1| 

**Note**: 
- U, S, and H represent the accuracy for unseen classes, seen classes, and their harmonic mean, respectively.
  
### CZSL

| Models | AWA1 | AWA2 | CUB | FLO |
| --- | --- | --- | --- | --- |
| GXE | 70.9 | 71.1 | 54.4 | - |
| LFGAA | - |68.1 | 67.6 | - |
| LisGAN | 70.6 |- | 58.8 | 69.6 |
| f-CLSWGAN | 68.2 |- | 57.3 | 67.2 | 
| f-VAEGAN-D2 | - |71.1 | 61.0 | 67.7 | 
| TF-VAEGAN |- | 72.2 | 64.9 | 70.8 |
| CE-GZSL | 71.0 | 70.4 | 77.5 | 70.6 |
| AGZSL | - |73.8 | 57.2 | 82.7 | 
| ESZSL | 58.2 | 58.6 | 53.9 | 51.1 |
| CMC-GAN | 71.4 |- | 61.4 | 69.8 | 
| SDGZSL | - |72.1 | 75.5 | 85.4 |
| Ours | 72.5 | 74.2 | 77.7 | 91.9 |
