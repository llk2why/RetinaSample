import os
import cv2
import yaml
import numpy as np

from config import Dataset

raw_dir = Dataset.RAW_DIR
energy_dict = {}

tifs = [x for x in os.listdir(raw_dir) if 'tif' in x.lower()]

for tif in tifs:
    fpath = os.path.join(raw_dir,tif)
    img = cv2.imread(fpath).astype(np.float)
    energy = float(np.sqrt(np.mean(np.square(img))))
    energy_dict[tif.split('.')[0]] = energy

with open('yamls/energy.yaml','w') as f:
    yaml.dump(energy_dict,f)
