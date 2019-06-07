import torch.utils.data as data
import numpy as np
import cv2

from os import listdir
from os.path import join
from PIL import Image

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
    def __init__(self, train_dir,target_dir,model_type=None,noisy=0.0, input_transform=None, target_transform=None,debug=False):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [x for x in listdir(train_dir) if is_image_file(x)]
        if debug:
            self.filenames = self.filenames[:400]
        self.image_train_filenames = [join(train_dir, x) for x in self.filenames]
        self.image_target_filenames = [join(target_dir, x) for x in self.filenames]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.model_type = model_type
        self.noisy = noisy

    def __getitem__(self, index):
        input = load_img(self.image_train_filenames[index]).astype(np.float)
        target = load_img(self.image_target_filenames[index]).astype(np.float)
        if self.noisy>0.0:
            self.__add_noisy__(input)
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        if self.model_type == 'RYYB':
            input = input
            r,g = input[:,:,0],input[:,:,1]
            g[g>0],r[g>0] = r[g>0]+g[g>0],0

    
        return input, target

    def __add_noisy__(self,x):
        noisy = np.random.normal(0,self.noisy,(x.shape))
        x = x+noisy

    def __len__(self):
        return len(self.image_train_filenames)
