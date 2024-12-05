
import os
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '4'
os.system('''python ...\\train.py --dataset AWA2 --ga 0.05 --nSample 5000 --gpu 1 --lr 0.00003 \
  --classifier_lr 0.001 --kl_warmup 0.01  --vae_dec_drop 0.5 --vae_enc_drop 0.4 \
  --gen_nepoch 420 --evl_start 200 --evl_interval 400  --manualSeed 6152''')
