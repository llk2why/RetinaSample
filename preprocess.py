# -*- coding=utf8 -*-
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

    def chop(img, suffix):
        r, c, _ = img.shape
        stride = PATCH_STRIDE
        cnt = 0
        patches = []
        irange = list(range(0, r - PATCH_SIZE, stride))
        jrange = list(range(0, c - PATCH_SIZE, stride))
        if irange[-1] + PATCH_SIZE != r:
            irange.append(r - PATCH_SIZE)
        if jrange[-1] + PATCH_SIZE != c:
            jrange.append(c - PATCH_SIZE)
        for i in irange:
            for j in jrange:
                fpath = os.path.join(Dataset.CHOPPED_DIR, '{}_{}{}'.format(name, cnt, suffix))
                if os.path.exists(fpath):
                    continue
                im = img[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
                cv2.imwrite(fpath, im)
                patches.append([[i, j], [i + PATCH_SIZE, j + PATCH_SIZE]])
                cnt += 1
                assert im.shape == (PATCH_SIZE, PATCH_SIZE, 3), '{}  {}'.format(im.shape, PATCH_SIZE)

        return patches

    print('Chopping images...')
    for i, (tif, fpath) in enumerate(zip(tifs, fpaths)):
        name, suffix = os.path.splitext(tif)
        img = cv2.imread(fpath)
        patches = chop(img, suffix)
        chop_info[name] = {'{}_{}'.format(name, i): patch for i, patch in enumerate(patches)}
        chop_info[name]['size'] = list(img.shape)
        print('\r{:.2f}%'.format(100.0 * i / len(tifs)), end='')
    print('\r100.0%')

    with open(YAML.CHOP_PATCH, 'w') as f:
        yaml.dump(chop_info, f)


def sample_bayer_imgs():
    if not os.path.exists(Dataset.MOSAIC_DIR):
        os.makedirs(Dataset.MOSAIC_DIR)
    if not os.path.exists(Dataset.MOSAIC_DIR_TEST):
        os.makedirs(Dataset.MOSAIC_DIR_TEST)
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR, pic)
        fpath_mosaic = os.path.join(Dataset.MOSAIC_DIR, pic)
        # if os.path.exists(fpath_mosaic):
        #     continue
        img = cv2.imread(fpath)
        im = np.stack([np.where(SAMPLE_MATRIX == i, img[:, :, i], 0) for i in range(3)], axis=-1)
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')

    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR_TEST) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR_TEST, pic)
        fpath_mosaic = os.path.join(Dataset.MOSAIC_DIR_TEST, pic)
        # if os.path.exists(fpath_mosaic):
        #     continue
        img = cv2.imread(fpath)
        im = np.stack([np.where(SAMPLE_MATRIX == i, img[:, :, i], 0) for i in range(3)], axis=-1)
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')


def splittest():
    if not os.path.exists(Dataset.CHOPPED_DIR_TEST):
        os.makedirs(Dataset.CHOPPED_DIR_TEST)
    if not os.path.exists(Dataset.MOSAIC_DIR_TEST):
        os.makedirs(Dataset.MOSAIC_DIR_TEST)

    def move(des, src):
        TIFs = [x for x in os.listdir(src) if 'TIF' in x]
        old_paths = [os.path.join(src, x) for x in TIFs]
        new_paths = [os.path.join(des, x) for x in TIFs]
        for old_path, new_path in zip(old_paths, new_paths):
            shutil.move(old_path, new_path)

    move(Dataset.CHOPPED_DIR_TEST, Dataset.CHOPPED_DIR)
    move(Dataset.MOSAIC_DIR_TEST, Dataset.MOSAIC_DIR)


def sample_ryyb_imgs():
    if not os.path.exists(Dataset.RYYB_MOSAIC_DIR):
        os.makedirs(Dataset.RYYB_MOSAIC_DIR)
    if not os.path.exists(Dataset.RYYB_MOSAIC_DIR_TEST):
        os.makedirs(Dataset.RYYB_MOSAIC_DIR_TEST)
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR, pic)
        fpath_mosaic = os.path.join(Dataset.RYYB_MOSAIC_DIR, pic)
        img = cv2.imread(fpath)
        im = img * RYYB_SAMPLE_MATRIX
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR_TEST) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR_TEST, pic)
        fpath_mosaic = os.path.join(Dataset.RYYB_MOSAIC_DIR_TEST, pic)
        img = cv2.imread(fpath)
        im = img * RYYB_SAMPLE_MATRIX
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')


def sample_random_imgs():
    if not os.path.exists(Dataset.Random_MOSAIC_DIR):
        os.makedirs(Dataset.Random_MOSAIC_DIR)
    if not os.path.exists(Dataset.Random_MOSAIC_DIR_TEST):
        os.makedirs(Dataset.Random_MOSAIC_DIR_TEST)
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR, pic)
        fpath_mosaic = os.path.join(Dataset.Random_MOSAIC_DIR, pic)
        img = cv2.imread(fpath)
        im = img * RANDOM_SAMPLE_MATRIX
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR_TEST) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR_TEST, pic)
        fpath_mosaic = os.path.join(Dataset.Random_MOSAIC_DIR_TEST, pic)
        img = cv2.imread(fpath)
        im = img * RANDOM_SAMPLE_MATRIX
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')


def sample_arbitrary_imgs():
    if not os.path.exists(Dataset.Arbitrary_MOSAIC_DIR):
        os.makedirs(Dataset.Arbitrary_MOSAIC_DIR)
    if not os.path.exists(Dataset.Arbitrary_MOSAIC_DIR_TEST):
        os.makedirs(Dataset.Arbitrary_MOSAIC_DIR_TEST)
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR) if 'tif' in x.lower()]
    r, c = PATCH_SIZE, PATCH_SIZE
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR, pic)
        fpath_mosaic = os.path.join(Dataset.Arbitrary_MOSAIC_DIR, pic)
        img = cv2.imread(fpath)
        # im = img*RANDOM_SAMPLE_MATRIX
        cfa = np.random.randint(0, 2, (r, c))
        sample = np.zeros_like(img)
        for j in range(3):
            channel = sample[:, :, j]
            channel[cfa == j] = 1
        im = img * sample
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR_TEST) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR_TEST, pic)
        fpath_mosaic = os.path.join(Dataset.Arbitrary_MOSAIC_DIR_TEST, pic)
        img = cv2.imread(fpath)
        # im = img*RANDOM_SAMPLE_MATRIX
        cfa = np.random.randint(0, 2, (r, c))
        sample = np.zeros_like(img)
        for j in range(3):
            channel = sample[:, :, j]
            channel[cfa == j] = 1
        im = img * sample
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')


def sample_rb_g_imgs():
    if not os.path.exists(Dataset.RB_G_MOSAIC_DIR):
        os.makedirs(Dataset.RB_G_MOSAIC_DIR)
    if not os.path.exists(Dataset.RB_G_MOSAIC_DIR_TEST):
        os.makedirs(Dataset.RB_G_MOSAIC_DIR_TEST)
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR, pic)
        fpath_mosaic = os.path.join(Dataset.RB_G_MOSAIC_DIR, pic)
        img = cv2.imread(fpath)
        im = img * RB_G_SAMPLE_MATRIX
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR_TEST) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR_TEST, pic)
        fpath_mosaic = os.path.join(Dataset.RB_G_MOSAIC_DIR_TEST, pic)
        img = cv2.imread(fpath)
        im = img * RB_G_SAMPLE_MATRIX
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')

def sample_jointpixel_rgbg_imgs():
    if not os.path.exists(Dataset.JointPixel_RGBG_MOSAIC_DIR):
        os.makedirs(Dataset.JointPixel_RGBG_MOSAIC_DIR)
    if not os.path.exists(Dataset.JointPixel_RGBG_MOSAIC_DIR_TEST):
        os.makedirs(Dataset.JointPixel_RGBG_MOSAIC_DIR_TEST)
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR, pic)
        fpath_mosaic = os.path.join(Dataset.JointPixel_RGBG_MOSAIC_DIR, pic)
        img = cv2.imread(fpath)
        im = np.stack([np.where(JOINTPIXEL_RGBG_MATRIX == i, img[:, :, i], 0) for i in range(3)], axis=-1)
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR_TEST) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR_TEST, pic)
        fpath_mosaic = os.path.join(Dataset.JointPixel_RGBG_MOSAIC_DIR_TEST, pic)
        img = cv2.imread(fpath)
        im = np.stack([np.where(JOINTPIXEL_RGBG_MATRIX == i, img[:, :, i], 0) for i in range(3)], axis=-1)
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')

def sample_jointpixel_triple_imgs():
    if not os.path.exists(Dataset.JointPixel_Triple_MOSAIC_DIR):
        os.makedirs(Dataset.JointPixel_Triple_MOSAIC_DIR)
    if not os.path.exists(Dataset.JointPixel_Triple_MOSAIC_DIR_TEST):
        os.makedirs(Dataset.JointPixel_Triple_MOSAIC_DIR_TEST)
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR, pic)
        fpath_mosaic = os.path.join(Dataset.JointPixel_Triple_MOSAIC_DIR, pic)
        img = cv2.imread(fpath)
        im = np.stack([np.where(JOINTPIXEL_Triple_MATRIX == i, img[:, :, i], 0) for i in range(3)], axis=-1)
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR_TEST) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR_TEST, pic)
        fpath_mosaic = os.path.join(Dataset.JointPixel_Triple_MOSAIC_DIR_TEST, pic)
        img = cv2.imread(fpath)
        im = np.stack([np.where(JOINTPIXEL_Triple_MATRIX == i, img[:, :, i], 0) for i in range(3)], axis=-1)
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')

def sample_jointpixel_sr_imgs():
    if not os.path.exists(Dataset.JointPixel_SR_MOSAIC_DIR):
        os.makedirs(Dataset.JointPixel_SR_MOSAIC_DIR)
    if not os.path.exists(Dataset.JointPixel_SR_MOSAIC_DIR_TEST):
        os.makedirs(Dataset.JointPixel_SR_MOSAIC_DIR_TEST)
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR, pic)
        fpath_mosaic = os.path.join(Dataset.JointPixel_SR_MOSAIC_DIR, pic)
        img = cv2.imread(fpath)
        im = np.stack([np.where(JOINTPIXEL_SR_MATRIX == i, img[:, :, i], 0) for i in range(3)], axis=-1)
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')
    pics = [x for x in os.listdir(Dataset.CHOPPED_DIR_TEST) if 'tif' in x.lower()]
    for i, pic in enumerate(pics):
        fpath = os.path.join(Dataset.CHOPPED_DIR_TEST, pic)
        fpath_mosaic = os.path.join(Dataset.JointPixel_SR_MOSAIC_DIR_TEST, pic)
        img = cv2.imread(fpath)
        im = np.stack([np.where(JOINTPIXEL_SR_MATRIX == i, img[:, :, i], 0) for i in range(3)], axis=-1)
        cv2.imwrite(fpath_mosaic, im)
        print('\r{:.2f}%'.format(100.0 * i / len(pics)), end='')
    print('\r100.0% ')


def start():
    # gather_info()
    # crop_imgs()
    # splittest()
    # sample_bayer_imgs()
    # sample_ryyb_imgs()
    # sample_random_imgs()
    # sample_arbitrary_imgs()
    # sample_rb_g_imgs()
    # sample_jointpixel_rgbg_imgs()
    # sample_jointpixel_triple_imgs()
    sample_jointpixel_sr_imgs()


if __name__ == '__main__':
    start()
