# -*- coding: utf-8 -*-
# @Author: JacobShi777

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import random
import argparse
import random
import functools
import time

from torch.autograd import Variable
from data import *
from model import *
import option
from myutils import utils
from myutils.vgg16 import Vgg16
from myutils.lcnn import LCNN
from myutils.Unet2 import *
import net
import torchvision.utils as vutils

import utils.fid_score

opt = option.init()

torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.myGpu


def train(print_every=10):
    checkpaths(opt)

    train_set = DatasetFromFolder(opt, True)
    test_set = DatasetFromFolder(opt, False)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    norm_layer = get_norm_layer(norm_type='batch')

    # D is Discriminator: Bx12x256x256 -> Bx1x131x131
    netD = NLayerDiscriminator(opt.input_nc, opt.ndf, n_layers=1, norm_layer=norm_layer, use_sigmoid=False, gpu_ids=opt.gpu_ids)
    # G is actually Face Encoder and Decoder: Bx1x256x256 -> Bx3x256x256
    netG = MyUnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=False, gpu_ids=opt.gpu_ids)
    # E is actually Composition Encoder: Bx8x256x256 -> [t1 ... t8]
    netE = MyEncoder(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=False, gpu_ids=opt.gpu_ids)

    VGG = make_encoder(model_file=opt.model_vgg)
    perceptual_loss = PerceptualLoss(VGG, 3)

    VGG.cuda()
    netG.cuda()
    netD.cuda()
    netE.cuda()
    
    # init all weights for Conv and BatchNorm2d
    netG.apply(weights_init)
    netD.apply(weights_init)
    netE.apply(weights_init)
    
    criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan)
    criterionL1 = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_E = torch.optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    print('=========== Networks initialized ============')
    print_network(netG)
    print_network(netD)
    print('=============================================')

    f = open('./Checkpoint/loss.txt', 'w')
    strat_time = time.time()
    G_epoch_loss = 0.0
    for epoch in range(1, opt.n_epoch + 1):
        fstr = f'Epoch: {epoch:>3d}'
        print(fstr)
        f.write(fstr+'\n')
        
        D_running_loss = 0.0
        G_running_loss = 0.0
        G1_running_loss = 0.0
        G2_running_loss = 0.0
        G3_running_loss = 0.0

        for (i, batch) in enumerate(training_data_loader, 1):
            # print(f'    Batch: {i:>5d}')
            real_p, real_s, identity = Variable(batch[0]), Variable(batch[1]), Variable(batch[2].squeeze(1))
            location = batch[3]

            real_p, real_s, identity = real_p.cuda(), real_s.cuda(), identity.cuda()

            # Discriminator BEGIN
            optimizer_D.zero_grad()
            # fake
            parsing_feature = netE.forward(real_p[:, 1:, :, :])
            fake_s = netG.forward(real_p[:, 0:1, :, :], parsing_feature)
            fake_ps = torch.cat((fake_s, real_p), 1)
            pred_fake = netD.forward(fake_ps.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # real
            real_ps = torch.cat((real_s, real_p), 1)
            pred_real = netD.forward(real_ps)
            loss_D_real = criterionGAN(pred_real, True)
            # step
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            # Discriminator END

            # Generator BEGIN
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            # Loss adv * 1
            pred_fake = netD.forward(fake_ps)
            loss_G_GAN = criterionGAN(pred_fake, True)
            # Loss cmp * lambda * ( alpha*local + (1-alpha)*global )
            loss_global = criterionL1(fake_s, real_s)
            loss_local = localLossL1(fake_s, real_s, real_p, criterionL1)
            loss_G_L1 = opt.alpha1 * loss_global + (1 - opt.alpha1) * loss_local
            loss_G_L1 *= opt.lambda1
            # Loss vgg * gamma
            b, c, w, h = fake_s.shape
            yh = fake_s.expand(b, 3, w, h)
            ys = real_s.expand(b, 3, w, h)
            # _mean = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).expand_as(yh)).cuda()
            # _var = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).expand_as(yh)).cuda()
            # yh = yh / 2 + 0.5
            # ys = ys / 2 + 0.5
            # yh = (yh - _mean) / _var
            # ys = (ys - _mean) / _var
            loss_recog = perceptual_loss(yh, ys)
            loss_recog *= opt.styleParam
            # Loss total
            loss_G = loss_G_GAN + loss_G_L1 + loss_recog
            # step
            loss_G.backward()
            optimizer_G.step()
            optimizer_E.step()
            # Generator END
            '''======================================================================='''

            D_running_loss += loss_D.item()
            G_running_loss += loss_G.item()
            G1_running_loss += loss_G_GAN.item()
            G2_running_loss += loss_G_L1.item()
            G3_running_loss += loss_recog.item()
            G_epoch_loss += loss_G.item()
            # print every 10 batch
            if i % print_every == 0:
                end_time = time.time()
                time_delta = usedtime(strat_time, end_time)
                fstr = (f'[{time_delta}] Batch: {i+1:>3d}; D loss: {D_running_loss/print_every:>7.5f};'  +
                                                        f' G loss: {G_running_loss/print_every:>7.5f};' +
                                                        f' G1 loss: {G1_running_loss/print_every:>7.5f};' +
                                                        f' G2 loss: {G2_running_loss/print_every:>7.5f};' +
                                                        f' G3 loss: {G3_running_loss/print_every:>7.5f};')
                print(fstr)
                f.write(fstr+'\n')
                
                D_running_loss = 0.0
                G_running_loss = 0.0
                G1_running_loss = 0.0
                G2_running_loss = 0.0
                G3_running_loss = 0.0

        fstr = (f'Epoch: {epoch:>3d}; Total loss: {G_epoch_loss/len(training_data_loader):>7.5f};')
        print(fstr)
        f.write(fstr+'\n')
        f.flush()
        
        G_epoch_loss = 0.0
        # if epoch >= 500 and epoch % 50 == 0:
        if epoch % 50 == 0:
            test(epoch, netG, netE, testing_data_loader, opt)
            checkpoint(epoch, netD, netG, netE)
            fid = utils.fid_score.get_fid([f'./Output/{epoch}',
                ['../Datasets/CUFS/AR/photos', '../Datasets/CUFS/CUHK/photos', '../Datasets/CUFS/XM2VTS/photos']])
            fstr = (f'Epoch: {epoch:>3d}; FID: {fid:>9.5f};')
            print(fstr)
            f.write(fstr+'\n')
            f.flush()
    f.close()


def test(epoch, netG, netE, test_data, opt):

    mkdir(opt.output)
    save_dir_A = opt.output + "/"+str(epoch)
    mkdir(save_dir_A)

    for i, batch in enumerate(test_data):
        real_p, real_s, identity = Variable(batch[0]), Variable(batch[1]), Variable(batch[2].squeeze(1))
        if opt.cuda:
            real_p, real_s, identity = real_p.cuda(), real_s.cuda(), identity.cuda()

        parsing_feature = netE(real_p[:, 1:, :, :])
        fake_s1 = netG.forward(real_p[:, 0:1, :, :], parsing_feature)
        output_name_A = '{:s}/{:s}{:s}'.format(
            save_dir_A, str(i + 1), '.jpg')
        temp = fake_s1[:, :, 3:253, 28:228].detach()
        vutils.save_image(temp, output_name_A, normalize=True, scale_each=True)

    print(str(epoch) + " saved")


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    train()
