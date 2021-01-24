
import os
import numpy as np
import glob
import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.functional as F
import torch.autograd as autograd
import math
from model import *
from dataset import *
from util import *
import argparse

class Train:
    
    def __init__(self, args):

        self.lr_g = args.lr_g
        self.lr_d = args.lr_d
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.triplet_margin = args.triplet_margin
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        
        self.train_data_dir = args.train_data_dir
        self.train_continue = args.train_continue
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        
        train_dataset = Dataset(data_dir=self.train_data_dir)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        
        tps_transformation = tps.TPS_SpatialTransformerNetwork(F=10, 
                                                               I_size=(256, 256), 
                                                               I_r_size=(256, 256), 
                                                               I_channel_num=3)
        
        # 모델 선언
        generator = Generator(sketch_channels=1, reference_channels=3, LR_negative_slope=0.2)
        discriminator = Discriminator(ndf=16, nChannels=4)
        
        # optimizer
        optimG = torch.optim.Adam(generator.parameters(), lr=self.lr_g, betas=(self.beta1, self.beta2))
        optimD = torch.optim.Adam(discriminator.parameters(), lr=self.lr_d, betas=(self.beta1, self.beta2))

        # model parameter 초기화
        init_net(generator, init_type='normal', init_gain=0.02, gpu_ids=[])
        init_net(discriminator, init_type='normal', init_gain=0.02, gpu_ids=[])
        
        # loss 함수 선언
        adversarial_loss = torch.nn.MSELoss()
        l1_loss = torch.nn.L1Loss()
        vgg_loss = VGGPerceptualLoss() #style loss + perceptual_loss

        for epoch in range(1,self.num_epoch+1):
            if epoch > 100:
                optimG, optimD = self.schedule_optim(optimG, optimD, epoch)
            for index, [appearance_img, sketch_img] in enumerate(train_loader):
                
                appearance_img = appearance_img.float()
                sketch_img = sketch_img.float()
                reference_img = tps_transformation(appearance_img)
                
                # ---------------------
                #  Train Generator
                # ---------------------
                fake_I_gt, quary, key, value = generator(sketch_img,reference_img)
                fake_output = discriminator(torch.cat((fake_I_gt,sketch_img), dim=1))
                g_adversarial_loss = adversarial_loss(fake_output,torch.ones_like(fake_output))
                g_l1_loss = l1_loss(fake_I_gt, appearance_img)
                g_triplet_loss = self.similarity_based_triple_loss(quary, key, value)
                g_perceptual_loss, g_style_loss = vgg_loss(appearance_img, fake_I_gt)
                
                g_loss = g_l1_loss + g_triplet_loss + g_adversarial_loss + g_style_loss + g_perceptual_loss
                
                optimG.zero_grad()
                g_loss.backward()
                optimG.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                fake_output = discriminator(torch.cat((fake_I_gt.detach(),sketch_img), dim=1))
                real_output = discriminator(torch.cat((appearance_img,sketch_img), dim=1))
                d_real_loss = adversarial_loss(real_output, torch.ones_like(real_output))
                d_fake_loss = adversarial_loss(fake_output, torch.zeros_like(fake_output))
                d_loss = d_real_loss+d_fake_loss
                
                optimD.zero_grad()
                d_loss.backward()
                optimD.step()
                
        
    def schedule_optim(self, optimG, optimD, epoch):
        optimG.param_groups[0]['lr'] = self.lr_g - (self.lr_g / (self.num_epoch - 100))*(epoch-100)
        optimD.param_groups[0]['lr'] = self.lr_d - (self.lr_d / (self.num_epoch - 100))*(epoch-100)
                
        return optimG, optimD
    
    def similarity_based_triple_loss(self, anchor, positive, negative):
        distance = self.scaled_dot_product(anchor, positive) - self.scaled_dot_product(anchor, negative) + self.triplet_margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss
    
    # https://www.quantumdl.com/entry/11%EC%A3%BC%EC%B0%A82-Attention-is-All-You-Need-Transformer
    def scaled_dot_product(self, query, key, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        return scores
    
        
