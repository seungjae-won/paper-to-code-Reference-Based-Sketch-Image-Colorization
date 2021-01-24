import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np
import torch
import torch.nn as nn
import math

class Generator(nn.Module):
    
    def __init__(self, sketch_channels=1, reference_channels=3, LR_negative_slope=0.2):
        super(Generator, self).__init__()
        
        self.encoder_sketch = Encoder(sketch_channels)
        self.encoder_reference = Encoder(reference_channels)
        self.scft = SCFT(sketch_channels, reference_channels)
        self.resblock = ResBlock(992, 992)
        self.unet_decoder = UNetDecoder()
    
    def forward(self, sketch_img, reference_img):
        
        # encoder 
        Vs, F = self.encoder_sketch(sketch_img)
        Vr, _ = self.encoder_reference(reference_img)
        
        # scft
        c, quary, key, value = self.scft(Vs,Vr)
        
        # resblock
        c_out = self.resblock(c)
        
        # unet decoder
        I_gt = self.unet_decoder(torch.cat((c,c_out),dim=1), F)

        return I_gt, quary, key, value

class Encoder(nn.Module):
    
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        
        def CL2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, LR_negative_slope=0.2):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)]
            layers += [nn.LeakyReLU(LR_negative_slope)]
            cbr = nn.Sequential(*layers)
            return cbr
        
        # conv_layer
        self.conv1 = CL2d(in_channels,16)
        self.conv2 = CL2d(16,16)
        self.conv3 = CL2d(16,32,stride=2)
        self.conv4 = CL2d(32,32)
        self.conv5 = CL2d(32,64,stride=2)
        self.conv6 = CL2d(64,64)
        self.conv7 = CL2d(64,128,stride=2)
        self.conv8 = CL2d(128,128)
        self.conv9 = CL2d(128,256,stride=2)
        self.conv10 = CL2d(256,256)
        
        # downsample_layer
        self.downsample1 = nn.AvgPool2d(kernel_size=16, stride=16)
        self.downsample2 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.downsample3 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.downsample4 = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):

        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)
        f6 = self.conv6(f5)
        f7 = self.conv7(f6)
        f8 = self.conv8(f7)
        f9 = self.conv9(f8)
        f10 = self.conv10(f9)
        
        F = [f9, f8, f7, f6, f5, f4, f3, f2 ,f1]
        
        v1 = self.downsample1(f1)
        v2 = self.downsample1(f2)
        v3 = self.downsample2(f3)
        v4 = self.downsample2(f4)
        v5 = self.downsample3(f5)
        v6 = self.downsample3(f6)
        v7 = self.downsample4(f7)
        v8 = self.downsample4(f8)

        V = torch.cat((v1,v2,v3,v4,v5,v6,v7,v8,f9,f10), dim=1)
        V = torch.reshape(V,(V.size(0),V.size(1),V.size(2)*V.size(3)))
        V = torch.transpose(V,1,2)
        
        return V,F

class SCFT(nn.Module):
    
    def __init__(self, sketch_channels, reference_channels, dv=992):
        super(SCFT, self).__init__()
        
        self.dv = torch.tensor(dv).float()
        
        self.w_q = nn.Linear(dv,dv)
        self.w_k = nn.Linear(dv,dv)
        self.w_v = nn.Linear(dv,dv)
        
    def forward(self, Vs, Vr):

        quary = self.w_q(Vs)
        key = self.w_k(Vr)
        value = self.w_v(Vr)

        c = torch.add(self.scaled_dot_product(quary,key,value), Vs)
        
        c = torch.transpose(c,1,2)
        c = torch.reshape(c,(c.size(0),c.size(1),16,16))
        
        return c, quary, key, value

    # https://www.quantumdl.com/entry/11%EC%A3%BC%EC%B0%A82-Attention-is-All-You-Need-Transformer
    def scaled_dot_product(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value)

class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        

        self.dec5_1 = CBR2d(in_channels=992+992, out_channels=256)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=512+128, out_channels=128)
        self.dec4_1 = CBR2d(in_channels=128+128, out_channels=128)
        self.unpool3 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=128+64, out_channels=64)
        self.dec3_1 = CBR2d(in_channels=64+64, out_channels=64)
        self.unpool2 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=64+32, out_channels=32)
        self.dec2_1 = CBR2d(in_channels=32+32, out_channels=32)
        self.unpool1 = nn.ConvTranspose2d(in_channels=32, out_channels=32,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=32+16, out_channels=16)
        self.dec1_1 = CBR2d(in_channels=16+16, out_channels=16)

        self.fc = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, F):
        
        dec5_1 = self.dec5_1(x)
        unpool4 = self.unpool4(torch.cat((dec5_1,F[0]),dim=1))

        dec4_2 = self.dec4_2(torch.cat((unpool4,F[1]),dim=1))
        dec4_1 = self.dec4_1(torch.cat((dec4_2,F[2]),dim=1))
        unpool3 = self.unpool3(dec4_1)

        dec3_2 = self.dec3_2(torch.cat((unpool3,F[3]),dim=1))
        dec3_1 = self.dec3_1(torch.cat((dec3_2,F[4]),dim=1))
        unpool2 = self.unpool2(dec3_1)

        dec2_2 = self.dec2_2(torch.cat((unpool2,F[5]),dim=1))
        dec2_1 = self.dec2_1(torch.cat((dec2_2,F[6]),dim=1))
        unpool1 = self.unpool1(dec2_1)
        
        dec1_2 = self.dec1_2(torch.cat((unpool1,F[7]),dim=1))
        dec1_1 = self.dec1_1(torch.cat((dec1_2, F[8]),dim=1))

        x = self.fc(dec1_1)
        
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        def block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]

            cbr = nn.Sequential(*layers)

            return cbr
        
        self.block_1 = block(in_channels,out_channels)
        self.block_2 = block(out_channels,out_channels)
        self.block_3 = block(out_channels,out_channels)
        self.block_4 = block(out_channels,out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        # block 1
        residual = x
        out = self.block_1(x)
        out += residual
        out = self.relu(out)
        
        # block 2
        residual = out
        out = self.block_2(x)
        out += residual
        out = self.relu(out)
        
        # block 3
        residual = out
        out = self.block_3(x)
        out += residual
        out = self.relu(out)
        
        # block 4
        residual = out
        out = self.block_4(x)
        out += residual
        out = self.relu(out)
        
        return out

# https://github.com/meliketoy/LSGAN.pytorch/blob/master/networks/Discriminator.py
# LSGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, ndf, nChannels):
        super(Discriminator, self).__init__()
        # input : (batch * nChannels * image width * image height)
        # Discriminator will be consisted with a series of convolution networks

        self.layer1 = nn.Sequential(
            # Input size : input image with dimension (nChannels)*64*64
            # Output size: output feature vector with (ndf)*32*32
            nn.Conv2d(
                in_channels = nChannels,
                out_channels = ndf,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            # Input size : input feature vector with (ndf)*32*32
            # Output size: output feature vector with (ndf*2)*16*16
            nn.Conv2d(
                in_channels = ndf,
                out_channels = ndf*2,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            # Input size : input feature vector with (ndf*2)*16*16
            # Output size: output feature vector with (ndf*4)*8*8
            nn.Conv2d(
                in_channels = ndf*2,
                out_channels = ndf*4,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            # Input size : input feature vector with (ndf*4)*8*8
            # Output size: output feature vector with (ndf*8)*4*4
            nn.Conv2d(
                in_channels = ndf*4,
                out_channels = ndf*8,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(
            # Input size : input feature vector with (ndf*8)*4*4
            # Output size: output probability of fake/real image
            nn.Conv2d(
                in_channels = ndf*8,
                out_channels = 1,
                kernel_size = 4,
                stride = 1,
                padding = 0,
                bias = False
            ),
            # nn.Sigmoid() -- Replaced with Least Square Loss
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out


"""
Copyright (c) 2019-present NAVER Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TPS_SpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border', align_corners = False)

        return batch_I_r


class LocalizationNetwork(nn.Module):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):
        """
        input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
        return batch_C_prime


class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # F+3 x F+3
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # n x F+3

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(
            # batch_size, 3, 2).float().cuda()), dim=1)  # batch_size x F+3 x 2
            batch_size, 3, 2).float()), dim=1)  # batch_size x F+3 x 2
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2