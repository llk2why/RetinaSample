import torch.utils.data as data
import torch
import yaml
import numpy as np
import cv2

from os import listdir
from os.path import join
from PIL import Image
from config import PATCH_SIZE

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tif", ".TIF"])

def load_img(filepath):
    # img = np.array(Image.open(filepath)).transpose(2,0,1)
    # img = np.array(Image.open(filepath))
    # img = Image.open(filepath).convert('RGB')
    # img = Image.open(filepath).convert('RGB')
    img = cv2.imread(filepath)
    
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, train_dir,target_dir,model_type=None,noise=0.0, input_transform=None, target_transform=None,debug=False):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [x for x in listdir(train_dir) if is_image_file(x)]
        if debug:
            self.filenames = self.filenames[:400]
        self.image_train_filenames = [join(train_dir, x) for x in self.filenames]
        self.image_target_filenames = [join(target_dir, x) for x in self.filenames]

        prefixes = ['_'.join(x.split('_')[:-1]) for x in self.filenames]

        with open('yamls/energy.yaml') as f:
            energy_dict = yaml.load(f,Loader=yaml.FullLoader)
        energy_dict = {k:v/255 for k,v in energy_dict.items()}
        self.avg_energy = [energy_dict[x] for x in prefixes]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.model_type = model_type
        self.noise = noise

        self.num = PATCH_SIZE*PATCH_SIZE*3

    def __getitem__(self, index):
        input = load_img(self.image_train_filenames[index])
        target = load_img(self.image_target_filenames[index])
        # if self.model_type == 'RYYB':
        #     r,g = input[:,:,0],input[:,:,1]
        #     g[g>0] = r[g>0]+g[g>0]
        #     r[g>0] = np.zeros_like(r[g>0])
        #     input[:,:,0],input[:,:,1] = r,g

        # ATTENTION, ToTensor will change dimension order and value range!
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        if self.model_type == 'RYYB':
            r = input[0,:,:]
            g = input[1,:,:]
            g[g>0] = r[g>0]+g[g>0]
            r[g>0] = torch.zeros(g[g>0].shape)
            input[0,:,:],input[1,:,:] = r,g
        if self.noise>0.0:
            input = self.__add_noise__(input,target,self.avg_energy[index])
            # input = self.__add_noise2__(input,target,self.avg_energy[index]) 
    
        return input, target


    def __add_noise__(self,x,y,avg_energy):
        # avg_energy = torch.sqrt(torch.sum(torch.pow(y.float(),2))/self.num)
        std = avg_energy*self.noise*torch.ones(x.shape)
        mu = torch.zeros(x.shape)
        e = torch.normal(mu,std)
        # Attension: the shape of image is (channel,row,col)
        if self.model_type in ['JointPixel_RGBG']: 
            e[:,1::2] = e[:,::2]/2
            e[:,::2] = e[:,::2]/2
        elif self.model_type in ['JointPixel_Triple']:
            e[:,:,:2] = e[:,:,:2]/3
            e[:,1::3] = e[:,::3]
            e[:,2::3] = e[:,::3]

        x = x+e*(x>0).float()
        x[x>1]=1.0
        x[x<0]=0.0
        return x

    def __add_noise2__(self,x,y,avg_energy):
        # avg_energy = torch.sqrt(torch.sum(torch.pow(y.float(),2))/self.num)
        std = avg_energy*self.noise*torch.ones(x.shape)
        mu = torch.zeros(x.shape)+10/255
        e = torch.normal(mu,std)
        # Attension: the shape of image is (channel,row,col)
        if self.model_type in ['JointPixel_RGBG']:
            e[:,1::2] = e[:,::2]/2
            e[:,::2] = e[:,::2]/2
        x = x+e*(x>0).float()
        x[x>1]=1.0
        x[x<0]=0.0
        return x

    def __len__(self):
        return len(self.image_train_filenames)
