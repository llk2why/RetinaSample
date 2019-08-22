import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.autograd import Variable
from collections import OrderedDict
from config import PATCH_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, size=256):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(in_channels=size, out_channels=size,
                             kernel_size=3, stride=1, padding=1, bias=True)),
            ('relu1', nn.PReLU()),
            ('c2', nn.Conv2d(in_channels=size, out_channels=size,
                             kernel_size=3, stride=1, padding=1, bias=True)),
        ]))
        self.shortcut = nn.Sequential()
        self.activate = nn.PReLU()

    def forward(self, input):
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
def get_padding(input, output, kernel_size, stride):
    padding = ((output - 1) * stride + kernel_size - input) // 2
    return padding


class DemosaicSR(nn.Module):
    def __init__(self, resnet_level=2):
        super(DemosaicSR, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ', nn.Conv2d(in_channels=1, out_channels=256,
                                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ', nn.PixelShuffle(2)),
            ('stage1_2_conv4x4', nn.Conv2d(in_channels=64, out_channels=256,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU', nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ', nn.Conv2d(in_channels=256, out_channels=256,
                                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU', nn.PReLU()),
            ('stage3_3_conv3x3', nn.Conv2d(in_channels=256, out_channels=3,
                                           kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        self.shortcut = nn.Sequential()

    def forward(self, input):
        output = torch.sum(input, dim=1, keepdim=True)
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = output + self.shortcut(input)
        return output

# TODO:
#  Special residual path for RYYB model
class RYYB(nn.Module):
    def __init__(self, resnet_level=2):
        super(RYYB, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ', nn.Conv2d(in_channels=3, out_channels=256,
                                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ', nn.PixelShuffle(2)),
            ('stage1_2_conv4x4', nn.Conv2d(in_channels=64, out_channels=256,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU', nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ', nn.Conv2d(in_channels=256, out_channels=256,
                                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU', nn.PReLU()),
            ('stage3_3_conv3x3', nn.Conv2d(in_channels=256, out_channels=3,
                                           kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        self.shortcut = nn.Sequential()

    def forward(self, input):
        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output)
        # output = output + self.shortcut(output)
        return output


class Random(nn.Module):
    def __init__(self, resnet_level=2):
        super(Random, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ', nn.Conv2d(in_channels=3, out_channels=256,
                                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ', nn.PixelShuffle(2)),
            ('stage1_2_conv4x4', nn.Conv2d(in_channels=64, out_channels=256,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU', nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ', nn.Conv2d(in_channels=256, out_channels=256,
                                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU', nn.PReLU()),
            ('stage3_3_conv3x3', nn.Conv2d(in_channels=256, out_channels=3,
                                           kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        self.shortcut = nn.Sequential()

    def forward(self, input):
        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output)
        output = output + self.shortcut(input)
        return output


class Arbitrary(nn.Module):
    def __init__(self, resnet_level=2):
        super(Arbitrary, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ', nn.Conv2d(in_channels=3, out_channels=256,
                                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ', nn.PixelShuffle(2)),
            ('stage1_2_conv4x4', nn.Conv2d(in_channels=64, out_channels=256,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU', nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ', nn.Conv2d(in_channels=256, out_channels=256,
                                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU', nn.PReLU()),
            ('stage3_3_conv3x3', nn.Conv2d(in_channels=256, out_channels=3,
                                           kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        self.shortcut = nn.Sequential()

    def forward(self, input):
        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output)
        output = output + self.shortcut(input)
        return output


class RB_G(nn.Module):
    def __init__(self, resnet_level=2):
        super(RB_G, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ', nn.Conv2d(in_channels=3, out_channels=256,
                                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ', nn.PixelShuffle(2)),
            ('stage1_2_conv4x4', nn.Conv2d(in_channels=64, out_channels=256,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU', nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ', nn.Conv2d(in_channels=256, out_channels=256,
                                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU', nn.PReLU()),
            ('stage3_3_conv3x3', nn.Conv2d(in_channels=256, out_channels=3,
                                           kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        self.shortcut = nn.Sequential()

    def forward(self, input):
        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output)
        output = output + self.shortcut(input)
        return output


class RB_G_DENOISE(nn.Module):
    def __init__(self, resnet_level=2):
        super(RB_G_DENOISE, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ', nn.Conv2d(in_channels=3, out_channels=256,
                                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ', nn.PixelShuffle(2)),
            ('stage1_2_conv4x4', nn.Conv2d(in_channels=64, out_channels=256,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU', nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            ('stage3_2_conv3x3 ', nn.Conv2d(in_channels=256, out_channels=256,
                                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU', nn.PReLU()),
            ('stage3_3_conv3x3', nn.Conv2d(in_channels=256, out_channels=2,
                                           kernel_size=3, stride=1, padding=1, bias=True))
        ]))

        unfold_size = 64
        denoise = [nn.Conv2d(in_channels=1, out_channels=unfold_size, kernel_size=3,
                             stride=1, padding=1, bias=True)] \
                  + [ResidualBlock(size=unfold_size) for i in range(resnet_level)] \
                  + [nn.Conv2d(in_channels=unfold_size, out_channels=1, kernel_size=3,
                               stride=1, padding=1, bias=True)]
        self.denoise = nn.Sequential(*denoise)
        self.shortcut = nn.Sequential()

    def forward(self, input):
        output = self.stage1(input)
        output = self.stage2(output)
        RnB = self.stage3(output) + self.shortcut(input[:, [0, 2], :, :])
        green = self.denoise(input[:, 1:2, :, :]) + self.shortcut(input[:, 1:2, :, :])
        output = torch.cat([RnB[:, 0:1, :, :], green, RnB[:, 1:2, :, :]], dim=1)
        return output

class JointPixel_RGBG(nn.Module):
    def __init__(self, resnet_level=2):
        super(JointPixel_RGBG, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ', nn.Conv2d(in_channels=3, out_channels=256,
                                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ', nn.PixelShuffle(2)),
            ('stage1_2_conv4x4', nn.Conv2d(in_channels=64, out_channels=256,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU', nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ', nn.Conv2d(in_channels=256, out_channels=256,
                                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU', nn.PReLU()),
            ('stage3_3_conv3x3', nn.Conv2d(in_channels=256, out_channels=3,
                                           kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        self.shortcut = nn.Sequential()

    def forward(self, input):
        input[:, :, ::2] = (input[:, :, ::2] + input[:, :, 1::2]) / 2
        input[:, :, 1::2] = input[:, :, ::2]
        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output) + self.shortcut(input)
        return output

class JointPixel_Triple(nn.Module):
    def __init__(self, resnet_level=2):
        super(JointPixel_Triple, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ', nn.Conv2d(in_channels=3, out_channels=256,
                                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ', nn.PixelShuffle(2)),
            ('stage1_2_conv4x4', nn.Conv2d(in_channels=64, out_channels=256,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU', nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            # ('stage3_1_SP_conv ',nn.PixelShuffle(2)),
            ('stage3_2_conv3x3 ', nn.Conv2d(in_channels=256, out_channels=256,
                                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU', nn.PReLU()),
            ('stage3_3_conv3x3', nn.Conv2d(in_channels=256, out_channels=3,
                                           kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        self.shortcut = nn.Sequential()

    def forward(self, input):
        num,ch,row,col = input.shape
        backup = input>=(1/255)
        zeros = torch.zeros(num,ch,1,col).to(input.device)
        output = torch.cat((input,zeros),dim=2)
        output[:,:,::3] =  output[:,:,::3] + output[:,:,1::3] + output[:,:,2::3]
        output[:,:,::3,::2] =  (output[:,:,::3,::2] + output[:,:,::3,1::2])/3
        output[:,:,1::3] = output[:,:,::3]
        output[:,:,2::3] = output[:,:,::3]
        output = output[:,:,:-1,:] * backup.float()
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output) + self.shortcut(input)
        return output

# TODO:
#  Special residual path for RYYB-like model
class Paramized_RYYB(nn.Module):
    def __init__(self, resnet_level=2):
        super(Paramized_RYYB, self).__init__()

        self.stage1 = nn.Sequential(OrderedDict([
            ('stage1_1_conv4x4 ', nn.Conv2d(in_channels=3, out_channels=256,
                                            kernel_size=4, stride=2, padding=1, bias=True)),
            ('stage1_2_SP_conv ', nn.PixelShuffle(2)),
            ('stage1_2_conv4x4', nn.Conv2d(in_channels=64, out_channels=256,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage1_2_PReLU', nn.PReLU())
        ]))
        stage2 = [ResidualBlock() for i in range(resnet_level)]
        self.stage2 = nn.Sequential(*stage2)
        self.stage3 = nn.Sequential(OrderedDict([
            ('stage3_2_conv3x3 ', nn.Conv2d(in_channels=256, out_channels=256,
                                            kernel_size=3, stride=1, padding=1, bias=True)),
            ('stage3_2_PReLU', nn.PReLU()),
            ('stage3_3_conv3x3', nn.Conv2d(in_channels=256, out_channels=3,
                                           kernel_size=3, stride=1, padding=1, bias=True))
        ]))
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.shortcut = nn.Sequential()

    def forward(self, input):
        eps = 0.99/255
        index = input[:,1,:,:]>eps
        # input[:,1,:,:][index] = self.alpha*input[:,0,:,:][index]+self.beta*input[:,1,:,:][index]
        input[:,1,:,:][index] = input[:,0,:,:][index]+input[:,1,:,:][index]
        input[:,0,:,:][index].fill_(0)

        output = self.stage1(input)
        output = self.stage2(output)
        output = self.stage3(output)
        return output


#TODO: explore new model for joint Pixel
