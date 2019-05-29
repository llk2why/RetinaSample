import os
import cv2
import yaml
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

def img2npz():
    train_tif = [x for x in os.listdir(Dataset.MOSAIC_DIR) if 'tif' in x]
    test_tif = [x for x in os.listdir(Dataset.MOSAIC_DIR) if 'TIF' in x]
    
    train_x_fpath = [os.path.join(Dataset.MOSAIC_DIR,pic) for pic in train_tif]
    train_y_fpath = [os.path.join(Dataset.CHOPPED_DIR,pic) for pic in train_tif]
    test_x_fpath = [os.path.join(Dataset.MOSAIC_DIR,pic) for pic in test_tif]
    test_y_fpath = [os.path.join(Dataset.CHOPPED_DIR,pic) for pic in test_tif]

    train_x1_fpath = train_x_fpath[:len(train_x_fpath)//2]
    train_x2_fpath = train_x_fpath[len(train_x_fpath)//2:]
    train_y1_fpath = train_y_fpath[:len(train_y_fpath)//2]
    train_y2_fpath = train_y_fpath[len(train_y_fpath)//2:]

    def save_train_x1():
        train_x1 = np.array([cv2.imread(fpath).transpose(2,0,1) for fpath in train_x1_fpath])
        np.save('./data/train_x1.npy',train_x1)
    def save_train_y1():
        train_y1 = np.array([cv2.imread(fpath).transpose(2,0,1) for fpath in train_y1_fpath])
        np.save('./data/train_y1.npy',train_y1)
    def save_train_x2():
        train_x2 = np.array([cv2.imread(fpath).transpose(2,0,1) for fpath in train_x2_fpath])
        np.save('./data/train_x2.npy',train_x2)
    def save_train_y2():
        train_y2 = np.array([cv2.imread(fpath).transpose(2,0,1) for fpath in train_y2_fpath])
        np.save('./data/train_y2.npy',train_y2)
    def save_test_x():
        test_x = np.array([cv2.imread(fpath).transpose(2,0,1) for fpath in test_x_fpath])
        np.save('./data/test_x.npy',test_x)
    def save_test_y():
        test_y = np.array([cv2.imread(fpath).transpose(2,0,1) for fpath in test_y_fpath])
        np.save('./data/test_y.npy',test_y)

    save_train_x1()
    save_train_y1()
    save_train_x2()
    save_train_y2()
    save_test_x()
    save_test_y()

    with open('./data/train_name1.yaml','w') as f:
        yaml.dump(train_tif[:len(train_tif)//2],f)
    with open('./data/train_name2.yaml','w') as f:
        yaml.dump(train_tif[len(train_tif)//2:],f)
    with open('./data/test_name1.yaml','w') as f:
        yaml.dump(test_tif,f)

def npz2img():
    train_x1 = np.load('./data/train_x1.npy')
    train_y1 = np.load('./data/train_y1.npy')
    print(train_x1[0].shape)
    cv2.imshow('t',train_x1[0].transpose(1,2,0))
    cv2.waitKey(0) 
    cv2.imshow('t',train_y1[0].transpose(1,2,0))
    cv2.waitKey(0)


def start():
    gather_info()
    # crop_imgs()
    # sample_imgs()
    # img2npz()
    # npz2img()

if __name__ == '__main__':
    start()
