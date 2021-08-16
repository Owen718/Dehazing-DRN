from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from utils.misc import *
import pdb
import DehazeNet  as Net
from skimage.measure import compare_ssim
import cv2
import torchvision.models as models
import h5py
import torch.nn.functional as F
import numpy as np
import time
import math
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='pix2pix_val',  help='')
parser.add_argument('--valDataroot', required=False, default='./', help='path to val dataset')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--inputChannelSize', type=int, default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int, default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--net', default='./test_model/net_epoch_2.pth', help="the pretrained model for test)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
opt = parser.parse_args()

create_exp_dir(opt.exp)

opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, #opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='Train',
                          shuffle=False,
                          seed=opt.manualSeed)

ngf = opt.ngf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

net = Net.dehaze(inputChannelSize, outputChannelSize, ngf)
if opt.net != '':
  net.load_state_dict(torch.load(opt.net))
net.cuda()


index=0
iteration = 0

for b, data in enumerate(valDataloader, 0):
    input, target, trans, ato = data
    batch_size = target.size(0)
    target, input, trans, ato = Variable(target), Variable(input), Variable(trans), Variable(ato)
    target, input, trans, ato = target.float().cuda(), input.float().cuda(), trans.float().cuda(), ato.float().cuda()
    fine_dehaze, tran_hat, atp_hat, dehaze = net(input)

    dehazed_data = fine_dehaze.data

    iteration = iteration + 1

    index2 = 0
    directory = './test_result/dehazed/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(opt.valBatchSize):
        index = index + 1
        print(index)
        dehazed_img = dehazed_data[index2, :, :, :]
        dehazed_img = dehazed_img.cpu().numpy()
        dehazed_img = dehazed_img.transpose(1,2,0)
        #dehazed_img = dehazed_img[:,:,::-1]
        scipy.misc.imsave('./test_result/dehazed/' + str(index - 1) + '.png', dehazed_img)








