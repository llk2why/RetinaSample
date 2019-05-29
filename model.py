import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

class Reshape(nn.Module):
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

# output = (input - kernel_size)/stride + 1
def get_padding(input,output,kernel_size,stride):
    pass

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 3@256x256 => 64@256x256
        #           => 32@256x256
        #           => 1@256x256

        self.layers = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3,64,9,padding=get_padding(256,256,9,1))), 
            # ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('c2', nn.Conv2d(64,32,1,padding=get_padding(256,256,1,1))),
            # ('bn2', nn.BatchNorm2d(32)),
            ('relu2', nn.ReLU()),
            ('c3', nn.Conv2d(32,1,5,padding=get_padding(256,256,5,1))),
            # ('bn3', nn.BatchNorm2d(1)),
            ('relu3', nn.ReLU()),
        ]))

    def forward(self, input):
        output = self.layers(input)
        return output