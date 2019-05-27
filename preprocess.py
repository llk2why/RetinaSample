import os
import cv2
from config import *
from PIL import Image
from collections import Counter

raw_dir = Dataset.RAW_DIR
tifs = [x for x in os.listdir(raw_dir) if 'tif' in x]
fpaths = [os.path.join(raw_dir,tif) for tif in tifs]

resolution = list()
chop_info = dict()
for fpath in fpaths:
    img = cv2.imread(fpath)
    resolution.append(img.shape)

with open()