import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt

import random
import os
from scipy import misc
from Colorferet import Colorferet
from training import get_training_batch, get_testing_batch, train, train_scene_discriminator
import utils

import models.resnet_128 as resnet_models
import models.dcgan_128 as dcgan_models
import models.dcgan_unet_128 as dcgan_unet_models
import models.vgg_unet_128 as vgg_unet_models
import models.classifiers as classifiers


def get_training_batch():
    while True:
        for sequence in train_loader:
            sequence.transpose_(0, 1)
            sequence.transpose_(3, 4).transpose_(2, 3)
            batch = [Variable(x) for x in sequence]
            yield batch
            
            
def get_testing_batch():
    while True:
        for sequence in test_loader:
            sequence.transpose_(0, 1)
            sequence.transpose_(3, 4).transpose_(2, 3)
            batch = [Variable(x) for x in sequence]
            yield batch
            
def train(x):
    netEP.zero_grad()
    netEC.zero_grad()
    netD.zero_grad()

    x_c1 = x[0].float().cuda()
    x_c2 = x[1].float().cuda()
    x_p1 = x[2].float().cuda()
    x_p2 = x[3].float().cuda()

    h_c1 = netEC(x_c1)
    h_c2 = netEC(x_c2)[0].detach()
    h_p1 = netEP(x_p1) # used for scene discriminator
    h_p2 = netEP(x_p2).detach()


    # similarity loss: ||h_c1 - h_c2||
    sim_loss = mse_criterion(h_c1[0], h_c2)


    # reconstruction loss: ||D(h_c1, h_p1), x_p1|| 
    rec = netD([h_c1, h_p1])
    rec_loss = mse_criterion(rec, x_p1)

    # scene discriminator loss: maximize entropy of output
    target = torch.cuda.FloatTensor(batch_size, 1).fill_(0.5)
    out = netC([h_p1, h_p2])
    sd_loss = bce_criterion(out, Variable(target))

    # full loss
    loss = sim_loss + rec_loss + adverserial_loss_weight * sd_loss
    loss.backward()

    optimizerEC.step()
    optimizerEP.step()
    optimizerD.step()

    return sim_loss.data.cpu().numpy(), rec_loss.data.cpu().numpy() 

def train_scene_discriminator(x):
    netC.zero_grad()

    target = torch.cuda.FloatTensor(batch_size, 1)

    x1 = x[0].float().cuda()
    x2 = x[1].float().cuda()
    h_p1 = netEP(x1).detach()
    h_p2 = netEP(x2).detach()

    half = int(batch_size/2)
    rp = torch.randperm(half).cuda()
    h_p2[:half] = h_p2[rp]
    target[:half] = 1
    target[half:] = 0
    out = netC([h_p1, h_p2])
    bce = bce_criterion(out, Variable(target))

    bce.backward()
    optimizerC.step()

    acc = out[:half].gt(0.5).sum() + out[half:].le(0.5).sum()
    return bce.data.cpu().numpy(), acc.data.cpu().numpy()/batch_size