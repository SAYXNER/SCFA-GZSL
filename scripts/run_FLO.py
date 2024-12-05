
import os
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['OMP_NUM_THREADS'] = '4'
os.system('''python ...\\train.py --dataset FLO_EPGN --ga 5 --nSample 1500 --gpu 0 \
  --lr 0.0001  --classifier_lr 0.005 --kl_warmup 0.01 --gen_nepoch 800 \
  --vae_dec_drop 0.4 --vae_enc_drop 0.4 --ae_drop 0.2''') 

