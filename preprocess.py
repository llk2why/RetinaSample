import os
import cv2
import yaml
import numpy as np
from config import *
from PIL import Image
from collections import Counter

raw_dir = Dataset.RAW_DIR
tifs = [x for x in os.listdir(raw_dir) if 'tif' in x]
fpaths = [os.path.join(raw_dir, tif) for tif in tifs]


def gather_info():
    resolution = list()
    for fpath in fpaths:
        img = cv2.imread(fpath)
        resolution.append(img.shape)
    resolution = dict(Counter(resolution))
    with open('info.txt', 'w', encoding='utf-8') as f:
        f.write('分辨率统计：\n')
        for k, v in resolution.items():
            f.write('\t{}:{}张'.format(k, v))


def crop_imgs():
    if not os.path.exists(Dataset.CHOPPED_DIR):
        os.makedirs(Dataset.CHOPPED_DIR)
    chop_info = dict()

    def chop(img):
        r, c, _ = img.shape
        stride = PATCH_STRIDE
        cnt = 0
        patches = []
        irange = list(range(0, r - PATCH_SIZE, stride))
        jrange = list(range(0, c - PATCH_SIZE, stride))
        if irange[-1]+PATCH_SIZE != r:
            irange.append(r-PATCH_SIZE)
        if jrange[-1]+PATCH_SIZE != c:
            jrange.append(c-PATCH_SIZE)
        for i in irange:
            for j in jrange:
                im = img[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
                fpath = os.path.join(Dataset.CHOPPED_DIR, '{}_{}.jpg'.format(name, cnt))
                cv2.imwrite(fpath,im)
                patches.append([[i, j], [i + PATCH_SIZE, j + PATCH_SIZE]])
                cnt += 1
                assert im.shape == (PATCH_SIZE,PATCH_SIZE,3),'{}  {}'.format(im.shape,PATCH_SIZE)
        
        return patches

    for tif, fpath in zip(tifs, fpaths):
        name = tif.split('.tif')[0]
        img = cv2.imread(fpath)
        patches = chop(img)
        chop_info[name] = {'{}_{}'.format(name, i): patch for i, patch in enumerate(patches)}
        chop_info[name]['size'] = list(img.shape)

    with open(YAML.CHOP_PATCH, 'w') as f:
        yaml.dump(chop_info, f)

def generate_sample_matrix(shape):



def start():
    gather_info()
    crop_imgs()


if __name__ == '__main__':
    start()
