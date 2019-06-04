import torch.utils.data as data
import numpy as np
import cv2
import torch

from os import listdir
from os.path import join
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tif", ".TIF"])

def load_img(filepath):
    # img = np.array(Image.open(filepath)).transpose(2,0,1)
    # img = np.array(Image.open(filepath))
    img = cv2.imread(filepath).transpose(2,0,1)
    img = torch.tensor(img).float()
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, train_dir,target_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [x for x in listdir(train_dir) if is_image_file(x)]
        self.image_train_filenames = [join(train_dir, x) for x in self.filenames]
        self.image_target_filenames = [join(target_dir, x) for x in self.filenames]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_train_filenames[index])
        target = load_img(self.image_target_filenames[index])
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        
        # print(torch.max(input))
        # print(torch.max(target))
        return input, target

    def __len__(self):
        return len(self.filenames)

# img = np.array(Image.open(r'C:\data\dataset\Sandwich 0612 fullsize Chopped\H1_0.tif'))
# # img = cv2.imread(r'C:\data\dataset\Sandwich 0612 fullsize Chopped\SDIM1952_407.tif')
# cv2.imshow('t',img)
# cv2.waitKey(0)