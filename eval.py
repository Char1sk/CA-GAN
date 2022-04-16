import cv2
import torch
from torch.autograd import Variable
import numpy as np
import os
import torchvision.utils as vutils
from data import *
from model import *
import option
from torch.utils.data import DataLoader
from myutils.Unet2 import *

if __name__ == '__main__':
    opt = option.init()
    norm_layer = get_norm_layer(norm_type='batch')
    netG = MyUnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, \
                            use_dropout=False, gpu_ids=opt.gpu_ids)

    netE = MyEncoder(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, \
                    use_dropout=False, gpu_ids=opt.gpu_ids)
    fold = opt.test_epoch
    
    netE.load_state_dict(torch.load('./Checkpoint/1Pure/netE_epoch_700.weight'))
    netG.load_state_dict(torch.load('./Checkpoint/1Pure/netG_epoch_700.weight'))
    # netE.load_state_dict(torch.load('../Models/sketch2photo/netE1_epoch_700.weight'))
    # netG.load_state_dict(torch.load('../Models/sketch2photo/netG1_epoch_700.weight'))
    # netE.load_state_dict(torch.load('../Models/sketch2photo/netE_epoch_250.weight'))
    # netG.load_state_dict(torch.load('../Models/sketch2photo/netG_epoch_250.weight'))

    netE.cpu()
    netG.cpu()
    # netE.cuda()
    # netG.cuda()
    
    imgmat = load_inputs('../Datasets/CUFS-CAGAN/CUHK/sketches/00.png', '../Datasets/CUFS-CAGAN/CUHK/mat2/00.mat', opt, False)
    imgmat = torch.from_numpy(imgmat)

    # test_set = DatasetFromFolder(opt, False)

    # testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    save_dir_A = opt.output + "/" + fold
    if not os.path.exists(save_dir_A):
        os.makedirs(save_dir_A)
    
    imgmat = imgmat.cpu()
    imgmat = imgmat.unsqueeze(0)

    parsing_feature = netE(imgmat[:, 1:, :, :])
    fake_s1 = netG.forward(imgmat[:, 0:1, :, :], parsing_feature)
    
    output_name_A = '{:s}/{:s}{:s}'.format(save_dir_A, str(114514), '.jpg')
    vutils.save_image(fake_s1[:, :, 3:253, 28:228], output_name_A, normalize=True, scale_each=True)
    # for i, batch in enumerate(testing_data_loader):
    #     real_p, real_s, identity = Variable(batch[0]), Variable(batch[1]), Variable(batch[2].squeeze(1))
    #     real_p, real_s, identity = real_p.cuda(), real_s.cuda(), identity.cuda()

    #     parsing_feature = netE(real_p[:, 1:, :, :])
    #     fake_s1 = netG.forward(real_p[:, 0:1, :, :], parsing_feature)
        
    #     output_name_A = '{:s}/{:s}{:s}'.format(save_dir_A, str(i + 1), '.jpg')
    #     vutils.save_image(fake_s1[:, :, 3:253, 28:228], output_name_A, normalize=True, scale_each=True)

    print(output_name_A, "saved")
