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
        if self.model_type == 'RYYB':
            input = input
            r,g = input[:,:,0],input[:,:,1]
            g[g>0],r[g>0] = r[g>0]+g[g>0],0

        # ATTENTION, ToTensor will change dimension order and value range!
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        if self.noise>0.0:
            input = self.__add_noise__(input,target,self.avg_energy[index])
        # print('input',input[0,:6,:6])
        # exit()
    
        return input, target

    def __add_noise__(self,x,y,avg_energy):
        # avg_energy = torch.sqrt(torch.sum(torch.pow(y.float(),2))/self.num)
        std = avg_energy*self.noise*torch.ones(x.shape)
        mu = torch.zeros(x.shape)
        e = torch.normal(mu,std)
        x = x+e*(x>0).float()
        return x

    def __len__(self):
        return len(self.image_train_filenames)
