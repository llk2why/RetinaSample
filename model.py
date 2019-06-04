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

# output = (input +2*padding - kernel_size)/stride + 1
def get_padding(input,output,kernel_size,stride):
    padding = ((output-1)*stride+kernel_size-input)//2
    return padding

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 3@128x128 => 64@128x128
        #           => 32@128x128
        #           => 1@128x128

        self.layers = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3,64,9,padding=get_padding(128,128,9,1))), 
            # ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('c2', nn.Conv2d(64,32,1,padding=get_padding(128,128,1,1))),
            # ('bn2', nn.BatchNorm2d(32)),
            ('relu2', nn.ReLU()),
            ('c3', nn.Conv2d(32,3,5,padding=get_padding(128,128,5,1))),
            # ('bn3', nn.BatchNorm2d(1)),
            ('relu3', nn.ReLU()),
        ]))
        self.layers.apply(weights_init)

    def forward(self, input):
        output = self.layers(input)
        return output

class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        # 3@128x128 => 64@128x128
        #           => 32@128x128
        #           => 1@128x128
        self.layers = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3,64,9,padding=get_padding(128,128,9,1))), 
            ('relu1', nn.ReLU()),
            ('c2', nn.Conv2d(64,32,1,padding=get_padding(128,128,1,1))),
            ('relu2', nn.ReLU()),
            ('c3', nn.Conv2d(32,3,5,padding=get_padding(128,128,5,1))),
            ('relu3', nn.ReLU()),
        ]))
        self.layers.apply(weights_init)

    def forward(self, input):
        output = self.layers(input)
        # output = output
        return output