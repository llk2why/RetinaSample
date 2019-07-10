import os
import cv2
import yaml
import numpy as np
from config import *
from utilities import LOG_INFO

LOG_INFO('====> Begin reading yaml')
PATCH_INFO = {}

names = [x.split('.TIF')[0] for x in os.listdir(Dataset.RAW_DIR) if 'TIF' in x]
def get_psnr(x,y):
    x,y = x.astype(np.float),y.astype(np.float)
    mse = np.mean(np.square(x-y))
    psnr = 20*np.log10(255)-10*np.log10(mse)
    return psnr

def joint_patch(patch_dir,name,tag):
    files = [x for x in os.listdir(patch_dir) if name in x]
    fpaths = [os.path.join(patch_dir,file) for file in files]
    
    patch_info = PATCH_INFO[name]
    
    joint = np.zeros(patch_info['size']).astype(np.float)
    cnt = np.zeros(patch_info['size']).astype(np.float)
    for file,fpath in zip(files,fpaths):
        patch_name = os.path.splitext(file)[0]
        img = cv2.imread(fpath)
        (r1,c1),(r2,c2) = patch_info[patch_name]
        joint[r1:r2,c1:c2] += img
        cnt[r1:r2,c1:c2] += 1
    joint = joint/cnt
    joint[joint>255]=255
    joint[joint<0]=0
    joint_dir = 'joint({})'.format(tag)
    if not os.path.exists(joint_dir):
        os.makedirs(joint_dir)
    cv2.imwrite('{}/{}.tiff'.format(joint_dir,name),joint.astype(np.uint8))

def combine(patch_dir,tag):
    LOG_INFO('====> Begin joint patches of {}'.format(tag))
    
    for i,name in enumerate(names):
        LOG_INFO('====> {}/{}'.format(i+1,len(names)))
        joint_patch(patch_dir,name,tag)

def compare_psnr(img_dir,suffix,tag):
    
    files = [x for x in os.listdir(img_dir) if suffix in x]
    psnrs = []
    LOG_INFO('====> Begin measuring PSNR of {}'.format(tag))    
    with open('result/psnr_{}.txt'.format(tag),'w') as f:
        for i,file in enumerate(files):
            fpath = os.path.join(img_dir,file)
            if 's' in file:
                std_name = name_pair[os.path.splitext(file)[0].split('_')[0]]
            else:
                std_name = file.split('.')[0]
            std_fpath = os.path.join('C:\data\dataset\Sandwich 0612 fullsize',std_name+'.TIF')
            # print(fpath)
            # print(std_fpath)
            im_y = cv2.imread(fpath)
            im_x = cv2.imread(std_fpath)
            psnr = get_psnr(im_x,im_y)
            psnrs.append(psnr)
            f.write('{}:{}\n'.format(file,psnr))
            print("{:.2f}%\r".format(100*(i+1)/len(files)),end='')
        print('100%   ')
        f.write('avg:{:.4f}\n'.format(np.mean(psnrs)))

def main():
    global PATCH_INFO
    with open('./yamls/chop.yaml') as f:
        PATCH_INFO = yaml.load(f)
    # combine(Dataset.RESULT,'RGGB')
    # combine(Dataset.RYYB_RESULT,'RYYB')
    # combine(Dataset.Random_RESULT,'Random')
    # combine(Dataset.Arbitrary_RESULT,'Arbitrary')
    combine(Dataset.RB_G_RESULT,'RB_G')

    # combine(Dataset.RESULT+' noise=0.10','RGGB_noise')
    # combine(Dataset.RYYB_RESULT+' noise=0.10','RYYB_noise')
    # combine(Dataset.Random_RESULT+' noise=0.10','Random_noise')
    combine(Dataset.RB_G_RESULT+' noise=0.10','RB_G_noise')

    # compare_psnr(r'joint(RGGB)','tiff','RGGB')
    # compare_psnr(r'joint(RYYB)','tiff','RYYB')
    # compare_psnr(r'joint(Random)','tiff','Random')
    # compare_psnr(r'joint(Arbitrary)','tiff','Arbitrary')
    compare_psnr(r'joint(RB_G)','tiff','RB_G')

    # compare_psnr(r'joint(RGGB_noise)','tiff','RGGB_noise')
    # compare_psnr(r'joint(RYYB_noise)','tiff','RYYB_noise')
    # compare_psnr(r'joint(Random_noise)','tiff','Random_noise')
    # compare_psnr(r'joint(Random_noise)','tiff','Random_noise')
    compare_psnr(r'joint(RB_G_noise)','tiff','RB_G_noise')

    

if __name__ == '__main__':
    main()