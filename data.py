# -*- coding: utf-8 -*-
# @Author: JacobShi777

import cv2
import os
import random
import numpy as np
import torch
import random
import scipy.io as sio
# import cPickle
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data as data


def formnames(infofile, if_train):
    infofile = infofile[0] if if_train else infofile[1]
    res = []
    with open(infofile, 'r') as f:
        for line in f:
            line = line.strip()
            res.append(line)
    return res


def input_transform(if_train, opt):
    if if_train:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    return transform


def target_transform(if_train, opt):
    if if_train:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    return transform


def load_inputs(imgpath, matpath, opt, if_train):
    # load 1 sketch and 8 composition
    imgpath = os.path.join(opt.root, imgpath)
    matpath = os.path.join(opt.root, matpath)
    # img: H*W*1, numpy
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    # img: 1*H*W, tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    img = transform(img)
    # img: H*W*1, numpy
    img = np.transpose(img.numpy(), (1, 2, 0))
    # img_fl: (1+19)*H*W, numpy
    img_fl = mat_merge(img, matpath)
    img_fl = np.transpose(img_fl, (2, 0, 1))
    # padding and choose mat
    if if_train:
        # img_fl: (1+8)*286*286
        img_fl = zero_padding(img_fl, opt.loadSize, opt.loadSize - img_fl.shape[1], opt.loadSize - img_fl.shape[2])
        img_fl = mat_process(img_fl)
    else:
        # img_fl: (1+8)*256*256
        img_fl = zero_padding(img_fl, opt.fineSize, opt.fineSize - img_fl.shape[1], opt.fineSize - img_fl.shape[2])
        img_fl = mat_process(img_fl)
    # 1 sketch + 8 mat
    return img_fl


def load_targets(imgpath, opt, if_train):
    # load 1 photo
    imgpath = os.path.join(opt.root, imgpath)
    # img: H*W*3 (BGR) numpy
    img = cv2.imread(imgpath)
    # img: H*W*3 (RGB) numpy
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img: 3*H*W (RGB) numpy
    img = np.transpose(img, (2, 0, 1))
    # padding
    if if_train:
        # img: 3*286*286
        img = zero_padding(img, opt.loadSize, opt.loadSize - img.shape[1], opt.loadSize - img.shape[2])
    else:
        # img: 3*256*256
        img = zero_padding(img, opt.fineSize, opt.fineSize - img.shape[1], opt.fineSize - img.shape[2])
    # 3c photo
    return img


def mat_merge(img, matpath):
    # img: H*W*C
    facelabel = sio.loadmat(matpath)
    temp = facelabel['res_label']
    img = np.concatenate((img, temp), axis=2)
    # img: H*W*(C+8)
    return img


def zero_padding(img, size0, pad1, pad2):
    zero_padding = np.zeros((img.shape[0], size0, size0), dtype=np.float32)
    pad1 = pad1 // 2
    pad2 = pad2 // 2
    zero_padding[:, pad1:size0 - pad1, pad2:size0 - pad2] = img
    return zero_padding


def mat_process(img_fl):
    # img_fl: (1+19)*H*W
    img_fl = img_fl.astype(np.float32)
    img = img_fl[0:1, :, :]

    temp = img_fl[1:, :, :]

    l0 = temp[0, :, :]

    l1 = temp[1, :, :]

    l2 = temp[7, :, :] + temp[6, :, :]
    l2 = np.where(l2 > 1, 1, l2)

    l3 = temp[5, :, :] + temp[4, :, :]
    l3 = np.where(l3 > 1, 1, l3)

    l4 = temp[2, :, :]

    l5 = temp[11, :, :] + temp[12, :, :]
    l5 = np.where(l5 > 1, 1, l5)

    l6 = temp[10, :, :]

    l7 = temp[13, :, :]

    # merge
    img = np.concatenate((img, l0.reshape(1, l0.shape[0], l0.shape[1])), axis=0)
    img = np.concatenate((img, l1.reshape(1, l1.shape[0], l1.shape[1])), axis=0)
    img = np.concatenate((img, l2.reshape(1, l2.shape[0], l2.shape[1])), axis=0)
    img = np.concatenate((img, l3.reshape(1, l3.shape[0], l3.shape[1])), axis=0)
    img = np.concatenate((img, l4.reshape(1, l4.shape[0], l4.shape[1])), axis=0)
    img = np.concatenate((img, l5.reshape(1, l5.shape[0], l5.shape[1])), axis=0)
    img = np.concatenate((img, l6.reshape(1, l6.shape[0], l6.shape[1])), axis=0)
    img = np.concatenate((img, l7.reshape(1, l7.shape[0], l7.shape[1])), axis=0)

    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, opt, if_train):
        super(DatasetFromFolder, self).__init__()

        self.if_train = if_train
        self.opt = opt
        self.imgnames = formnames(opt.infofile, self.if_train)
        self.input_transform_train = input_transform(self.if_train, self.opt)
        self.target_transform_train = target_transform(self.if_train, self.opt)

    def __getitem__(self, index):

        imgname = self.imgnames[index]
        item = imgname.split('||')
        inputs = load_inputs(item[0], item[2], self.opt, self.if_train)
        targets = load_targets(item[1], self.opt, self.if_train)
        identity = torch.LongTensor([int(item[3])])
        if self.if_train:
            w_offset = random.randint(0, self.opt.loadSize - self.opt.fineSize - 1)
            h_offset = random.randint(0, self.opt.loadSize - self.opt.fineSize - 1)
            inputs = inputs[:, h_offset:h_offset + self.opt.fineSize, h_offset:h_offset + self.opt.fineSize]
            targets = targets[:, h_offset:h_offset + self.opt.fineSize, h_offset:h_offset + self.opt.fineSize]
            location = torch.LongTensor([int(h_offset), int(w_offset)])
        else:
            location = torch.LongTensor([15, 15])
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)

        return inputs, targets, identity, location

    def __len__(self):
        return len(self.imgnames)


def checkpaths(opt):
    if not os.path.exists(opt.checkpoint):
        os.mkdir(opt.checkpoint)
    if not os.path.exists(opt.gen_root):
        os.mkdir(opt.gen_root)


def checkpoint(epoch, netD, netG, netE):
    netD_out_path = "./Checkpoint/netD_epoch_{}.weight".format(epoch)
    torch.save(netD.state_dict(), netD_out_path)
    netG_out_path = "./Checkpoint/netG_epoch_{}.weight".format(epoch)
    torch.save(netG.state_dict(), netG_out_path)
    netE_out_path = "./Checkpoint/netE_epoch_{}.weight".format(epoch)
    torch.save(netE.state_dict(), netE_out_path)

    print("Checkpoint saved to {} and {}".format(netD_out_path, netG_out_path))


def usedtime(strat_time, end_time):
    delta = int(end_time - strat_time)
    hours = delta // 3600
    minutes = (delta - hours * 3600) // 60
    seconds = delta - hours * 3600 - minutes * 60
    # return ('%2d:%2d:%2d' % (hours, minutes, seconds))
    return (f'{hours:>02d}:{minutes:>02d}:{seconds:>02d}')
