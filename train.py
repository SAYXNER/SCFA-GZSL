import numpy
import torch
import json
import torch.optim as optim
import glob
import os
import random
import argparse
from time import gmtime, strftime

import losses
from models import *
from dataset_GBU_SD import FeatDataLayer, DATA_LOADER, map_label
from utils import *
import torch.backends.cudnn as cudnn
import classifier
from tqdm import trange
import torch.nn.functional as F
device = torch.device("cuda")
import torch.autograd as autograd
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2',help='dataset: CUB, AWA2, APY, FLO, SUN')
parser.add_argument('--dataroot', default='...', help='path to dataset')
# False 非直推式
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)
parser.add_argument('--gen_nepoch', type=int, default=600, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate to train generater')
parser.add_argument('--zsl', type=bool, default=True, help='Evaluate ZSL or GZSL')
parser.add_argument('--ga', type=float, default=3, help='relationNet weight')
parser.add_argument('--beta', type=float, default=0.3, help='tc weight')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--kl_warmup', type=float, default=0.002, help='kl warm-up for VAE')
parser.add_argument('--tc_warmup', type=float, default=0.001, help='tc warm-up')
parser.add_argument('--vae_dec_drop', type=float, default=0.5, help='dropout rate in the VAE decoder')
parser.add_argument('--vae_enc_drop', type=float, default=0.4, help='dropout rate in the VAE encoder')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--classifier_steps', type=int, default=20, help='training steps of the classifier')
parser.add_argument('--batchsize', type=int, default=256, help='input batch size')
parser.add_argument('--nSample', type=int, default=1200, help='number features to generate per class')
parser.add_argument('--disp_interval', type=int, default=400)
parser.add_argument('--save_interval', type=int, default=10000)#10000
parser.add_argument('--evl_interval',  type=int, default=500)
parser.add_argument('--evl_start',  type=int, default=100)
parser.add_argument('--manualSeed', type=int, default=3740, help='manual seed')

parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
parser.add_argument('--block_dim', type=int, default=144)
parser.add_argument('--finetune', type=bool, default=False, help='Use fine-tuned feature')

parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--gammaD', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--ae_drop', type=float, default=0.2, help='dropout rate in the auto-encoder')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')#0.5
parser.add_argument('--gpu', default='1', type=str, help='index of GPU to use')
parser.add_argument('--nghA', type=int, default=4096)
parser.add_argument('--nhF', type=int, default=2048)
parser.add_argument('--ins_temp', type=float, default=0.3)
parser.add_argument('--cls_weight', type=float, default=1)
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
opt.gpu = torch.device("cuda:"+opt.gpu if torch.cuda.is_available() else "cpu")

def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(opt.batchsize, 1)
    alpha = alpha.expand(real_data.size()).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data).cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size()).cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty
def train():
    dataset = DATA_LOADER(opt)
    if opt.zsl:
        out_dir = 'cVAE_contrast_Relation_center/{}/final/ZSL'.format(opt.dataset)
    else:
        out_dir = 'cVAE_contrast_Relation_center/{}/train/GZSL'.format(opt.dataset)

    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))
    log_dir = out_dir + '/all_log_lr-{}_clr-{}_nSample-{}_cls-{}_ga-{}.txt'.format(opt.lr, opt.classifier_lr, opt.nSample, opt.cls_weight, opt.ga)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
    dataset.feature_dim = dataset.train_feature.shape[1]
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class

    one = torch.tensor(1, dtype=torch.float).cuda()
    mone = one * -1
    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)
    opt.niter = int(dataset.ntrain/opt.batchsize) * opt.gen_nepoch

    result_gzsl_soft = Result()
    result_zsl_soft = Result()

    model = VAE(opt).cuda()
    ae = AE(opt).cuda()
    relationNet = RelationNet(opt).cuda()
    netDec = AttDec(opt).cuda()

    print(model)
    print(ae)

    print(relationNet)

    tr_cls_centroid_seen = np.zeros([dataset.seenclasses.shape[0], opt.X_dim], np.float16)
    for i in range(dataset.seenclasses.shape[0]):
        tr_cls_centroid_seen[i] = np.mean(
            dataset.train_feature[map_label(dataset.train_label, dataset.seenclasses) == i].cpu().detach().numpy(),
            axis=0)
    cent1 = torch.from_numpy(dataset.tr_cls_centroid).cuda()
    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('Generative Model Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,  betas=(opt.beta1, 0.999))
    ae_optimizer = optim.Adam(ae.parameters(), lr=opt.lr,  betas=(opt.beta1, 0.999))
    relation_optimizer = optim.Adam(relationNet.parameters(), lr=opt.lr,  betas=(opt.beta1, 0.999))
    netDec_optimizer = optim.Adam(netDec.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    mse = nn.MSELoss().cuda()
    iters = math.ceil(dataset.ntrain/opt.batchsize)
    beta = 0.01
    coin = 0
    gamma = 0
    gp_sum = 0
    for it in trange(start_step, opt.niter+1):
        if it % iters == 0:
            beta = min(opt.kl_warmup*(it/iters), 1)
            gamma = min(opt.tc_warmup * (it / iters), 1)

        blobs = data_layer.forward()  
        feat_data = blobs['data'] 
        labels_numpy = blobs['labels'].astype(int)  
        labels = torch.from_numpy(labels_numpy.astype('int')).cuda()

        contras_criterion = losses.SupConLoss_clear(opt.ins_temp)
        C = np.array([dataset.train_att[i, :] for i in labels])
        C = torch.from_numpy(C.astype('float32')).cuda()
        X = torch.from_numpy(feat_data).cuda()
        sample_C = torch.from_numpy(np.array([dataset.train_att[i, :] for i in labels.unique()])).cuda()
        sample_C_n = labels.unique().shape[0]
        sample_label = labels.unique().cuda()
        x_mean, z_mu, z_var, z = model(X, C)
        loss_cVAE, ce, kl = multinomial_loss_function(x_mean, X, z_mu, z_var, z, beta=beta)

        sample_labels = np.array(sample_label)
        re_batch_labels = []
        for label in labels_numpy:
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)
        one_hot_labels = torch.zeros(opt.batchsize, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()

        h1, x1 = ae(x_mean)

        relations = relationNet(h1, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])

        p_loss = opt.ga * mse(relations, one_hot_labels)

        h2, x2 = ae(X)

        relations = relationNet(h2, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])

        p_loss = p_loss + opt.ga * mse(relations, one_hot_labels)

        mu_real, var_real, att_dec_real = netDec(h2)
        real_att_contras_loss = contras_criterion(att_dec_real, labels)
        cos_dist = torch.einsum('bd,nd->bn', att_dec_real, C)
        CLS_loss = nn.CrossEntropyLoss()
        Latt_real = CLS_loss(cos_dist, labels.long())
        mu_real, var_real, att_dec_fake = netDec(h1)
        fake_att_contras_loss = contras_criterion(att_dec_fake, labels)
        cos_dist = torch.einsum('bd,nd->bn', att_dec_fake, C)
        CLS_loss = nn.CrossEntropyLoss()
        Latt_fake = CLS_loss(cos_dist, labels.long())
        errG = opt.cls_weight * (fake_att_contras_loss + real_att_contras_loss) + opt.cls_weight * (Latt_real + Latt_fake)
    

        rec = mse(x1, X) + mse(x2, X)


        center_loss1 = Variable(torch.Tensor([0.0])).cuda()
        for i in range(dataset.ntrain_class):
            sample_idx = (labels == i).data.nonzero().squeeze()
            if sample_idx.numel() == 0:
                center_loss1 += 0.0
            else:
                G_sample_cls = h2[sample_idx, :]
                center_loss1 += (G_sample_cls.mean(dim=0) - cent1[i]).pow(2).sum().sqrt()
        center_loss2 = Variable(torch.Tensor([0.0])).cuda()
        for i in range(dataset.ntrain_class):
            sample_idx = (labels == i).data.nonzero().squeeze()
            if sample_idx.numel() == 0:
                center_loss2 += 0.0
            else:
                G_sample_cls = h1[sample_idx, :]
                center_loss2 += (G_sample_cls.mean(dim=0) - cent1[i]).pow(
                    2).sum().sqrt()
        loss = loss_cVAE + p_loss + rec + errG + 0.01*(center_loss1 + center_loss2)

        optimizer.zero_grad()
        relation_optimizer.zero_grad()
        ae_optimizer.zero_grad()

        netDec_optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        relation_optimizer.step()
        ae_optimizer.step()
        netDec_optimizer.step()

        if it % opt.disp_interval == 0 and it:
            log_text = '{};Iter-[{}/{}]; loss: {:.3f}; loss_cVAE:{:.3f}; p_loss1:{:.3f}; rec:{:.3f}; fake_att_contras_loss:{:.3f};real_att_contras_loss:{:.3f};Latt_fake:{:.3f};Latt_real:{:.3f};center_loss1:{:.3f};center_loss2:{:.3f};'.format(opt.dataset,
                                    it, opt.niter, loss.item(), loss_cVAE.item(), p_loss.item(), rec.item(), fake_att_contras_loss.item(), real_att_contras_loss.item(), Latt_real.item(), Latt_fake.item(), center_loss1.item(), center_loss2.item())
            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it > opt.evl_start:

            model.eval()
            ae.eval()
            #gen_feat, gen_label = synthesize_feature_match_model(model, match_model, dataset, opt)
            gen_feat, gen_label = synthesize_feature_test(model, ae, dataset, opt)
            with torch.no_grad():
                train_feature = ae.encoder(dataset.train_feature.cuda()).cuda()
                test_unseen_feature = ae.encoder(dataset.test_unseen_feature.cuda()).cuda()
                test_seen_feature = ae.encoder(dataset.test_seen_feature.cuda()).cuda()

            train_X = torch.cat((train_feature, gen_feat.cuda()), 0)
            train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)
            """ZSL"""
            cls = classifier.CLASSIFIER(opt, gen_feat, gen_label, dataset, test_seen_feature, test_unseen_feature,
                                        dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5, 20,
                                        opt.nSample, False)
            result_zsl_soft.update(it, cls.acc)
            log_print("ZSL Softmax:", log_dir)
            log_print("Acc {:.2f}%  Best_acc [{:.2f}% | Iter-{}]".format(
                cls.acc, result_zsl_soft.best_acc, result_zsl_soft.best_iter), log_dir)

            if result_zsl_soft.save_model:
                files2remove = glob.glob(out_dir + '/Best_model_ZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                    """
                save_model_ae(it, model, opt.manualSeed, log_text,
                              out_dir + '/Best_model_ZSL_{}_ACC_{:.2f}.tar'.format(
                                  it, result_zsl_soft.best_acc)
                              )
"""
            #if opt.zsl:
                """ GZSL"""
            cls = classifier.CLASSIFIER(opt, train_X, train_Y, dataset, test_seen_feature, test_unseen_feature,
                                        dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5,
                                        opt.classifier_steps, opt.nSample, True)

            result_gzsl_soft.update_gzsl(it, cls.acc_seen, cls.acc_unseen, cls.H)

            log_print("GZSL Softmax:", log_dir)
            log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                cls.acc_unseen, cls.acc_seen, cls.H, result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

            if result_gzsl_soft.save_model:
                files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                    """
                    save_model_ae(it, model, opt.manualSeed, log_text,
                                  out_dir + '/Best_model_GZSL_{}_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(it,result_gzsl_soft.best_acc,result_gzsl_soft.best_acc_S_T,result_gzsl_soft.best_acc_U_T))
"""

            model.train()
            ae.train()
            # ae.train()
        if it % opt.save_interval == 0 and it:
            save_model(it, model, opt.manualSeed, log_text,
                       out_dir + '/Iter_{:d}.tar'.format(it))
            print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))

    print('Dataset', opt.dataset)
    #if opt.zsl:
    print("the best ZSL seen accuracy is",result_zsl_soft.best_acc)
    #else:
    print('the best GZSL seen accuracy is',result_gzsl_soft.best_acc_S_T)
    print('the best GZSL unseen accuracy is',result_gzsl_soft.best_acc_U_T)
    print('the best GZSL H is', result_gzsl_soft.best_acc)


if __name__ == "__main__":
    train()
