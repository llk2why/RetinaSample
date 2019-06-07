import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import numpy as np


from torch.autograd import Variable
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(OrderedDict([
            ('c1',nn.Conv2d(in_channels=256, out_channels=256, 
                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('relu1',nn.PReLU()),
            ('c2',nn.Conv2d(in_channels=256, out_channels=256, 
                            kernel_size=3, stride=1, padding=1, bias=True)),
        ]))
        self.shortcut = nn.Sequential()
        self.activate = nn.PReLU()
    
    def forward(self,input):
        output = self.left(input)
        output += self.shortcut(input)
        output = self.activate(output)
        return output

class Reshape(nn.Module):
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

# output = (input +2*padding - kernel_size)/stride + 1
def get_padding(input,output,kernel_size,stride):
    padding = ((output-1)*stride+kernel_size-input)//2
    return padding

class DemosaicSR(nn.Module):
    def __init__(self,resnet_level=2):
        super(DemosaicSR, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ',nn.Conv2d(in_channels=1, out_channels=256,
                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ',nn.PixelShuffle(2)),
            ('stage1_2_conv4x4',nn.Conv2d(in_channels=64, out_channels=256,
                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU',nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ',nn.Conv2d(in_channels=256, out_channels=256,
                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU',nn.PReLU()),
            ('stage3_3_conv3x3',nn.Conv2d(in_channels=256, out_channels=3,
                            kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        

    def forward(self, input):
        output = torch.sum(input,dim=1,keepdim=True)
        # print(output.shape)
        output = self.stage1(output)
        # print(output.shape)
        output = self.stage2(output)
        # print(output.shape)
        output = self.stage3(output)
        # print(output.shape)
        return output

class RYYB(nn.Module):
    def __init__(self,resnet_level=2):
        super(RYYB, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ',nn.Conv2d(in_channels=3, out_channels=256,
                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ',nn.PixelShuffle(2)),
            ('stage1_2_conv4x4',nn.Conv2d(in_channels=64, out_channels=256,
                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU',nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ',nn.Conv2d(in_channels=256, out_channels=256,
                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU',nn.PReLU()),
            ('stage3_3_conv3x3',nn.Conv2d(in_channels=256, out_channels=3,
                            kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        

    def forward(self, input):
        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output)
        return output

class Random(nn.Module):
    def __init__(self,resnet_level=2):
        super(Random, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ',nn.Conv2d(in_channels=3, out_channels=256,
                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ',nn.PixelShuffle(2)),
            ('stage1_2_conv4x4',nn.Conv2d(in_channels=64, out_channels=256,
                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU',nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ',nn.Conv2d(in_channels=256, out_channels=256,
                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU',nn.PReLU()),
            ('stage3_3_conv3x3',nn.Conv2d(in_channels=256, out_channels=3,
                            kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        

    def forward(self, input):
        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output)
        return output

class Arbitrary(nn.Module):
    def __init__(self,resnet_level=2):
        super(Arbitrary, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ',nn.Conv2d(in_channels=3, out_channels=256,
                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ',nn.PixelShuffle(2)),
            ('stage1_2_conv4x4',nn.Conv2d(in_channels=64, out_channels=256,
                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU',nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ',nn.Conv2d(in_channels=256, out_channels=256,
                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU',nn.PReLU()),
            ('stage3_3_conv3x3',nn.Conv2d(in_channels=256, out_channels=3,
                            kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        self.shortcut = nn.Sequential()
        

    def forward(self, input):
        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output)
        output = output + self.shortcut(input)
        return output