from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

import torch.nn.functional as F

def reparameter(mu, sigma):
    return (torch.randn_like(mu) * sigma) + mu

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.z_size = args.Z_dim
        self.args = args
        self.q_z_nn_output_dim = args.q_z_nn_output_dim
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()
        self.FloatTensor = torch.FloatTensor

    def create_encoder(self):
        q_z_nn = nn.Sequential(
            nn.Linear(self.args.X_dim + self.args.C_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.Dropout(self.args.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, self.q_z_nn_output_dim)
        )
        q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)

        q_z_var = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.z_size),
            nn.Dropout(0.2),
            nn.Softplus(),
        )
        return q_z_nn, q_z_mean, q_z_var


    def create_decoder(self):
        p_x_nn = nn.Sequential(
            nn.Linear(self.z_size + self.args.C_dim, 2048),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True)
        )
        p_x_mean = nn.Sequential(
            nn.Linear(2048, self.args.X_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return p_x_nn, p_x_mean


    def reparameterize(self, mu, var):
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_().cuda()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x, c):
        input = torch.cat((x,c),1)
        h = self.q_z_nn(input)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z, c):
        input = torch.cat((z, c), 1)
        h = self.p_x_nn(input)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x, c, weights=None):
        z_mu, z_var = self.encode(x, c)
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z, c)
        return x_mean, z_mu, z_var, z





class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.X_dim, args.X_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop)
        )

        self.decoder = nn.Sequential(
            nn.Linear(args.X_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
            nn.Linear(2048, args.X_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
        )

    def forward(self, x):
        z = self.encoder(x)
        x1 = self.decoder(z)
        return z, x1


class RelationNet(nn.Module):
    def __init__(self, args):
        super(RelationNet, self).__init__()
        self.fc1 = nn.Linear(args.C_dim + args.X_dim, 2048)
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, s, c):

        c_ext = c.unsqueeze(0).repeat(s.shape[0], 1, 1)
        cls_num = c_ext.shape[1]

        s_ext = torch.transpose(s.unsqueeze(0).repeat(cls_num, 1, 1), 0, 1)
        relation_pairs = torch.cat((s_ext, c_ext), 2).view(-1, c.shape[1] + s.shape[1])
        relation = nn.ReLU()(self.fc1(relation_pairs))
        relation = nn.Sigmoid()(self.fc2(relation))
        return relation



    # total correlation model 分辨hs和hu的无关性 通过重组排序的方法
class Dis_TC(nn.Module):
    def __init__(self, opt):
        super(Dis_TC, self).__init__()
        self.fc1 = nn.Linear(opt.S_dim+opt.S_dim, opt.nhF)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, hs, hu):
        h = torch.cat((hs, hu), dim=1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        h = self.lrelu(h)
        return h

class Dis_Embed_Att(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att, self).__init__()
        self.fc1 = nn.Linear(opt.X_dim+opt.C_dim, opt.nhF)
        self.fc2 = nn.Linear(opt.nhF, opt.C_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        #print('input.size'+str(input.size()))
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h

class Dis_Embed_Att1(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att1, self).__init__()
        self.fc1 = nn.Linear(opt.S_dim+opt.C_dim, opt.nhF)
        self.fc2 = nn.Linear(opt.nhF, opt.C_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        #print('input.size'+str(input.size()))
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h

class AttDec(nn.Module):
    def __init__(self, opt):
        super(AttDec, self).__init__()
        self.hidden = None
        self.attSize = opt.C_dim
        self.fc1 = nn.Linear(opt.X_dim, opt.nghA)
        self.fc3 = nn.Linear(opt.nghA, opt.C_dim * 2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, feat):
        h = feat
        h = self.lrelu(self.fc1(h))
        self.hidden = h
        h = self.fc3(h)
        mus, stds = h[:, :self.attSize], h[:, self.attSize:]
        stds = self.sigmoid(stds)
        h = reparameter(mus, stds)
        mus = F.normalize(mus, dim=1)
        h = h / h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))
        return mus, stds, h

    def getLayersOutDet(self):
        return self.hidden.detach()


    # total correlation model 分辨hs和hu的无关性 通过重组排序的方法
class Dis_TC(nn.Module):
    def __init__(self, opt):
        super(Dis_TC, self).__init__()
        self.fc1 = nn.Linear(opt.S_dim+opt.S_dim, opt.nhF)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, hs, hu):
        h = torch.cat((hs, hu), dim=1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        h = self.lrelu(h)
        return h
