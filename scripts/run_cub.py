
import os
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['OMP_NUM_THREADS'] = '4'
os.system('''python E:\GITHUB\cVAE_cons测\\train_no_center.py --dataset CUB_STC  --ga 3 --nSample 1200 --gpu 0 \
  --lr 0.0001  --classifier_lr 0.005 --gen_nepoch 550 --kl_warmup 0.02 --weight_decay 1e-6 \
  --vae_enc_drop 0.4 --vae_dec_drop 0.5 --ae_drop 0.2''')

