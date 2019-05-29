import os
import cv2
import yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt

from config import *
from PIL import Image
from collections import Counter

raw_dir = Dataset.RAW_DIR
tifs = [x for x in os.listdir(raw_dir) if 'tif' in x or 'TIF' in x]
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
            f.write('\t{}:{}张\n'.format(k, v))


def crop_imgs():
    if not os.path.exists(Dataset.CHOPPED_DIR):
        os.makedirs(Dataset.CHOPPED_DIR)
    chop_info = dict()

    def chop(img,suffix):
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
                fpath = os.path.join(Dataset.CHOPPED_DIR, '{}_{}{}'.format(name, cnt,suffix))
                cv2.imwrite(fpath,im)
                patches.append([[i, j], [i + PATCH_SIZE, j + PATCH_SIZE]])
                cnt += 1
                assert im.shape == (PATCH_SIZE,PATCH_SIZE,3),'{}  {}'.format(im.shape,PATCH_SIZE)
        
        return patches
    print('Chopping images...')
    for i,tif, fpath in enumerate(zip(tifs, fpaths)):
        name,suffix = os.path.splitext(tif)
        img = cv2.imread(fpath)
        patches = chop(img,suffix)
        chop_info[name] = {'{}_{}'.format(name, i): patch for i, patch in enumerate(patches)}
        chop_info[name]['size'] = list(img.shape)
        print('{}'.format(100.0*i/len(tif)),end='\r')
    print('100.0%')

    with open(YAML.CHOP_PATCH, 'w') as f:
        yaml.dump(chop_info, f)

def sample_imgs():
    if not os.path.exists(Dataset.MOSAIC_DIR):
        os.makedirs(Dataset.MOSAIC_DIR)
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR) if 'tif' in x.lower()]
    for i,pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR,pic)
        img = cv2.imread(fpath)
        
        im = np.stack([np.where(SAMPLE_MATRIX==i,img[:,:,i], 0) for i in range(3)],axis=-1)
        fpath_mosaic = os.path.join(Dataset.MOSAIC_DIR,pic)
        cv2.imwrite(fpath_mosaic,im)
        print('{}'.format(100.0*i/len(pics)),end='\r')
    print('100.0%')

def splittest():
    if not os.path.exists(Dataset.CHOPPED_DIR_TEST):
        os.makedirs(Dataset.CHOPPED_DIR_TEST)
    if not os.path.exists(Dataset.MOSAIC_DIR_TEST):
        os.makedirs(Dataset.MOSAIC_DIR_TEST)
    
    def move(des,src):
        TIFs = [x for x in os.listdir(src) if 'TIF' in x]
        old_paths = [os.path.join(src,x) for x in TIFs]
        new_paths = [os.path.join(des,x) for x in TIFs]
        for old_path,new_path in zip(old_paths,new_paths):
            shutil.move(old_path,new_path)
         
    move(Dataset.CHOPPED_DIR_TEST,Dataset.CHOPPED_DIR)
    move(Dataset.MOSAIC_DIR_TEST,Dataset.MOSAIC_DIR)
    

def start():
    # gather_info()
    crop_imgs()
    sample_imgs()
    splittest()

if __name__ == '__main__':
    start()
