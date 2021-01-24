#from google.colab import drive
#from torch.utils.tensorboard import SummaryWriter
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
from model import *
from util import *
import argparse
from torch.autograd import Variable
from train import *


# Parser 생성하기
parser = argparse.ArgumentParser(description='Reference-Based Sketch Image Colorization using Augmented-Self Reference and Dense Semantic Correspondence', 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr_g', default=1e-4, type=float, dest='lr_g')
parser.add_argument('--lr_d', default=2e-4, type=float, dest='lr_d')
parser.add_argument('--beta1', default=0.5, type=float, dest='beta1')
parser.add_argument('--beta2', default=0.999, type=float, dest='beta2')
parser.add_argument('--batch_size', default=1, type=int, dest='batch_size')
parser.add_argument('--triplet_margin', default=12, type=int, dest='triplet_margin')
parser.add_argument('--num_epoch', default=200, type=int, dest='num_epoch')

parser.add_argument('--train_data_dir', default='C:/Users/wonseungjae/Desktop/github/data/pokemon/pokemon', type=str, dest='train_data_dir')
parser.add_argument('--mode', default='train', type=str, dest='mode')
parser.add_argument('--train_continue', default='off', type=str, dest='train_continue')


PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.train()
    elif ARGS.mode == 'test':
        TRAINER.test()
    else:
        print('='*40)
        print('The entered "mode" does not exist')
        print('='*40)

if __name__ == '__main__':
    main()

