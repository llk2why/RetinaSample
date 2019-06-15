import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from config import *


def get_cfa_imgs(name):
    save_names = ['RGGB','RYYB','Random']
    dirs = [Dataset.MOSAIC_DIR_TEST,Dataset.RYYB_MOSAIC_DIR_TEST,Dataset.Random_MOSAIC_DIR_TEST]
    for save_name,dir_path in zip(save_names,dirs):
        img = cv2.imread(os.path.join(dir_path,name))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        half = 20
        img = img[64-half:64+half,64-half:64+half]
        img = np.repeat(img,8,axis=0)
        img = np.repeat(img,8,axis=1)
        plt.imsave('poster/cfa_{}.png'.format(save_name),img)
        # plt.imshow()
        # plt.show()
        # cv2.imshow('t',img)
        # cv2.waitKey(0)

def get_reconstruct_imgs(name):
    save_names = ['RGGB','RYYB','Random']
    dirs = [Dataset.RESULT,Dataset.RYYB_RESULT,Dataset.Random_RESULT]
    for save_name,dir_path in zip(save_names,dirs):
        img = cv2.imread(os.path.join(dir_path,name))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        half = 20
        img = img[64-half:64+half,64-half:64+half]
        img = np.repeat(img,8,axis=0)
        img = np.repeat(img,8,axis=1)
        plt.imsave('poster/re_{}.png'.format(save_name),img)
        # plt.show()
        # cv2.imshow('t',img)
        # cv2.waitKey(0)

def main():
    name = 'SDIM1905_668.tif'
    get_cfa_imgs(name)
    get_reconstruct_imgs(name)

if __name__ == '__main__':
    main()