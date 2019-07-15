import os
import sys
import cv2
import numpy as np

sys.path.append('..')
from config import *

raw_dir = Dataset.RAW_DIR

tifs = [x for x in os.listdir(raw_dir) if 'tif' in x.lower()]

template = np.tile(JOINTPIXEL_RGBG_MATRIX,(100,100))
r,c = template.shape
img = np.ones((r,c,3))
mask = np.stack([np.where(template == i, img[:, :, i], 0) for i in range(3)], axis=-1)
# cv2.imshow('a',mask*255)
# cv2.waitKey(0)

with open('neighbor_distance.txt','w') as f:
    for tif in tifs:
        fpath = os.path.join(raw_dir,tif)
        img = cv2.imread(fpath).astype(np.float)
        if img.shape[0]%2:
            a = img[::2][:-1]
            b = img[1::2]
        else:
            a,b = img[::2],img[1::2]
        dt = np.mean(np.abs(a-b))
        delta = np.mean(np.square((a-b)/2))
        f.write('{}\tabs:{}\tmse:{}\t-log10(mse):{}\n'.format(tif,dt,delta,-np.log10(delta)))

with open('neighbor_distance_mask.txt','w') as f:
    for tif in tifs:
        fpath = os.path.join(raw_dir,tif)
        img = cv2.imread(fpath).astype(np.float)
        if img.shape[0]%2:
            a = img[::2][:-1]
            b = img[1::2]
        else:
            a,b = img[::2],img[1::2]
        a = a*mask[:a.shape[0],:a.shape[1]]
        b = b*mask[:b.shape[0],:b.shape[1]]
        dt = np.mean(np.abs(a-b))
        delta = np.mean(np.square((a-b)/2))
        f.write('{}\tabs:{}\tmse:{}\t-log10(mse):{}\n'.format(tif,dt,delta,-np.log10(delta)))
    
    

