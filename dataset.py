import torch.utils.data as data
import numpy as np

from os import listdir
from os.path import join
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tif", ".TIF"])

def load_img(filepath):
    # img = np.array(Image.open(filepath)).transpose(2,0,1)
    # img = np.array(Image.open(filepath))
    img = Image.open(filepath).convert('RGB')
    
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, train_dir,target_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [x for x in listdir(train_dir) if is_image_file(x)][:200]
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

        return input, target

    def __len__(self):
        return len(self.image_train_filenames)
