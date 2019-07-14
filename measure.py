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
    joint_dir = 'joint/joint({})'.format(tag)
    if not os.path.exists(joint_dir):
        os.makedirs(joint_dir)
    cv2.imwrite('{}/{}.tiff'.format(joint_dir,name),joint.astype(np.uint8))

def combine(patch_dir,tag):
    LOG_INFO('====> Begin joint patches of {}'.format(tag))
    
    for i,name in enumerate(names):
        LOG_INFO('====> {}/{}'.format(i+1,len(names)))
        joint_patch(patch_dir,name,tag)

def compare_psnr(img_dir,suffix,tag):
    if not os.path.exists('result'):
        os.makedirs('result')
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
            std_fpath = os.path.join(Dataset.RAW_DIR,std_name+'.TIF')
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

def extract():
    txts = [x for x in os.listdir('result') if 'psnr' in x]
    with open('result/results.txt','w') as f:
        for txt in txts:
            fpath = os.path.join('result',txt)
            with open(fpath,'r') as fin:
                lines = fin.readlines()
                line = lines[18].strip()
                f.write('{}:{}\n'.format(txt[5:][:-4],line.split(':')[1]))
        

def main():
    # global PATCH_INFO
    # with open('./yamls/chop.yaml') as f:
    #     PATCH_INFO = yaml.load(f)
    # combine(Dataset.RESULT,'RGGB')
    # combine(Dataset.RYYB_RESULT,'RYYB')
    # combine(Dataset.Random_RESULT,'Random')
    # combine(Dataset.Arbitrary_RESULT,'Arbitrary')
    # combine(Dataset.RB_G_RESULT,'RB_G')
    # combine(Dataset.RB_G_DENOISE_RESULT,'RB_G_DENOISE')
    # combine(Dataset.JointPixel_RGBG_RESULT,'JointPixel_RGBG')

    # for val in ['0.05','0.10','0.20','0.50']:
    #     combine(Dataset.RESULT+' noise={}'.format(val),'RGGB_noise={}'.format(val))
    #     combine(Dataset.RYYB_RESULT+' noise={}'.format(val),'RYYB_noise={}'.format(val))
    #     combine(Dataset.Random_RESULT+' noise={}'.format(val),'Random_noise={}'.format(val))
    #     combine(Dataset.RB_G_RESULT+' noise={}'.format(val),'RB_G_noise={}'.format(val))
    #     combine(Dataset.RB_G_DENOISE_RESULT+' noise={}'.format(val),'RB_G_DENOISE_noise={}'.format(val))
    #     combine(Dataset.JointPixel_RGBG_RESULT+' noise={}'.format(val),'JointPixel_RGBG_noise={}'.format(val))

    compare_psnr(r'joint/joint(RGGB)','tiff','RGGB')
    compare_psnr(r'joint/joint(RYYB)','tiff','RYYB')
    compare_psnr(r'joint/joint(Random)','tiff','Random')
    compare_psnr(r'joint/joint(Arbitrary)','tiff','Arbitrary')
    compare_psnr(r'joint/joint(RB_G)','tiff','RB_G')
    compare_psnr(r'joint/joint(RB_G_DENOISE)','tiff','RB_G_DENOISE')
    compare_psnr(r'joint/joint(JointPixel_RGBG)','tiff','JointPixel_RGBG')

    for val in ['0.05','0.10','0.20','0.50']:
        compare_psnr(r'joint/joint(RGGB_noise={})'.format(val),'tiff','RGGB_noise={}'.format(val))
        compare_psnr(r'joint/joint(RYYB_noise={})'.format(val),'tiff','RYYB_noise={}'.format(val))
        compare_psnr(r'joint/joint(Random_noise={})'.format(val),'tiff','Random_noise={}'.format(val))
        compare_psnr(r'joint/joint(RB_G_noise={})'.format(val),'tiff','RB_G_noise={}'.format(val))
        compare_psnr(r'joint/joint(RB_G_DENOISE_noise={})'.format(val),'tiff','RB_G_DENOISE_noise={}'.format(val))
        compare_psnr(r'joint/joint(JointPixel_RGBG_noise={})'.format(val),'tiff','JointPixel_RGBG_noise={}'.format(val))

    extract()

if __name__ == '__main__':
    main()