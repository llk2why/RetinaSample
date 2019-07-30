import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from config import *
from utilities import LOG_INFO
from collections import defaultdict

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
            im_y = cv2.imread(fpath)
            im_x = cv2.imread(std_fpath)
            psnr = get_psnr(im_x,im_y)
            psnrs.append(psnr)
            f.write('{}:{}\n'.format(file,psnr))
            print("{:.2f}%\r".format(100*(i+1)/len(files)),end='')
        print('100%   ')
        f.write('avg:{:.4f}\n'.format(np.mean(psnrs)))

def extract(dir_path):
    txts = [x for x in os.listdir(dir_path) if 'psnr' in x]
    with open(os.path.join(dir_path,'results.txt'),'w') as f:
        for txt in txts:
            fpath = os.path.join(dir_path,txt)
            with open(fpath,'r') as fin:
                lines = fin.readlines()
                line = lines[18].strip()
                f.write('{}:{}\n'.format(txt[5:][:-4],line.split(':')[1]))

def combine_switch():
    global PATCH_INFO
    LOG_INFO('====> Begin reading yaml')
    with open('./yamls/chop.yaml') as f:
        PATCH_INFO = yaml.load(f,Loader=yaml.FullLoader)
    # combine(Dataset.RESULT,'RGGB')
    combine(Dataset.RYYB_RESULT,'RYYB')
    # combine(Dataset.Random_RESULT,'Random')
    # combine(Dataset.Arbitrary_RESULT,'Arbitrary')
    # combine(Dataset.RB_G_RESULT,'RB_G')
    # combine(Dataset.RB_G_DENOISE_RESULT,'RB_G_DENOISE')
    # combine(Dataset.JointPixel_RGBG_RESULT,'JointPixel_RGBG')
    combine(Dataset.JointPixel_RGBG_RESULT,'JointPixel_Triple')
    # combine(Dataset.JointPixel_RGBG_RESULT,'Paramized_RYYB')

    for val in ['0.02','0.05','0.08','0.10','0.15','0.20','0.25','0.30','0.40','0.50']:
    # for val in ['0.05','0.20']:
    # for val in ['0.05']:
        # combine(Dataset.RESULT+' noise={}'.format(val),'RGGB_noise={}'.format(val))
        combine(Dataset.RYYB_RESULT+' noise={}'.format(val),'RYYB_noise={}'.format(val))
        # combine(Dataset.Random_RESULT+' noise={}'.format(val),'Random_noise={}'.format(val))
        # combine(Dataset.RB_G_RESULT+' noise={}'.format(val),'RB_G_noise={}'.format(val))
        # combine(Dataset.RB_G_DENOISE_RESULT+' noise={}'.format(val),'RB_G_DENOISE_noise={}'.format(val))
        # combine(Dataset.JointPixel_RGBG_RESULT+' noise={}'.format(val),'JointPixel_RGBG_noise={}'.format(val))
        combine(Dataset.JointPixel_Triple_RESULT+' noise={}'.format(val),'JointPixel_Triple_noise={}'.format(val))
        # combine(Dataset.Paramized_RYYB_RESULT+' noise={}'.format(val),'Paramized_RYYB_noise={}'.format(val))

def compare_psnr_switch():
    # compare_psnr(r'joint/joint(RGGB)','tiff','RGGB')
    compare_psnr(r'joint/joint(RYYB)','tiff','RYYB')
    # compare_psnr(r'joint/joint(Random)','tiff','Random')
    # compare_psnr(r'joint/joint(Arbitrary)','tiff','Arbitrary')
    # compare_psnr(r'joint/joint(RB_G)','tiff','RB_G')
    # compare_psnr(r'joint/joint(RB_G_DENOISE)','tiff','RB_G_DENOISE')
    # compare_psnr(r'joint/joint(JointPixel_RGBG)','tiff','JointPixel_RGBG')
    compare_psnr(r'joint/joint(JointPixel_Triple)','tiff','JointPixel_Triple')
    # compare_psnr(r'joint/joint(Paramized_RYYB)','tiff','Paramized_RYYB')

    for val in ['0.02','0.05','0.08','0.10','0.15','0.20','0.25','0.30','0.40','0.50']:
    # for val in ['0.05','0.10','0.20','0.50']:
    # for val in ['0.05','0.20']:
    # for val in ['0.05']:
        # compare_psnr(r'joint/joint(RGGB_noise={})'.format(val),'tiff','RGGB_noise={}'.format(val))
        compare_psnr(r'joint/joint(RYYB_noise={})'.format(val),'tiff','RYYB_noise={}'.format(val))
        # compare_psnr(r'joint/joint(Random_noise={})'.format(val),'tiff','Random_noise={}'.format(val))
        # compare_psnr(r'joint/joint(RB_G_noise={})'.format(val),'tiff','RB_G_noise={}'.format(val))
        # compare_psnr(r'joint/joint(RB_G_DENOISE_noise={})'.format(val),'tiff','RB_G_DENOISE_noise={}'.format(val))
        # compare_psnr(r'joint/joint(JointPixel_RGBG_noise={})'.format(val),'tiff','JointPixel_RGBG_noise={}'.format(val))
        compare_psnr(r'joint/joint(JointPixel_Triple_noise={})'.format(val),'tiff','JointPixel_Tripl_noise={}'.format(val))
        # compare_psnr(r'joint/joint(Paramized_RYYB_noise={})'.format(val),'tiff','Paramized_RYYB_noise={}'.format(val))

def plot(dir_path,name):
    print(dir_path)
    with open(os.path.join(dir_path,'results.txt')) as f:
        lines = f.readlines()
    psnrs = {l.strip().split(':')[0]:float(l.strip().split(':')[1]) for l in lines}
    mapping = defaultdict(list)
    print(psnrs.keys())
    for cfa in ['RGGB','RYYB','Random','RB_G','RB_G_DENOISE','JointPixel_RGBG','JointPixel_Triple']:
    # for cfa in ['RGGB','RB_G','RB_G_DENOISE','JointPixel_RGBG']:
        # for noise in ['','0.05','0.10','0.20','0.50']:
        # for noise in ['0.05','0.20']:
        for noise in ['','0.02','0.05','0.08','0.10','0.15','0.20','0.25','0.30','0.40','0.50']:
            key = '{}_noise={}'.format(cfa,noise) if noise!='' else cfa
            mapping[cfa].append(psnrs[key])
    print(mapping)
    # noises = [0,0.05,0.10,0.20,0.50]
    # noises = [0.05,0.20]
    noises = [0,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50]
    # for cfa in ['RGGB','RB_G','RB_G_DENOISE','JointPixel_RGBG']:
    for cfa in ['RGGB','RYYB','Random','RB_G','RB_G_DENOISE','JointPixel_RGBG','JointPixel_Triple']:
        # print(cfa)
        # print(noises)
        # print(len(mapping[cfa]))
        plt.plot(noises,mapping[cfa],label=cfa)
    for noise in noises:
        plt.vlines(noises,ymin=28,ymax=49,color='gray',linestyles='dotted') 
    plt.legend()
    plt.xlim(-0.02,0.52)
    plt.ylim(28,50)
    plt.title('Demosaicking Performance')
    plt.xlabel('$\sigma coefficient$')
    plt.ylabel('PSNR/dB')
    plt.savefig(name)

def main():
    combine_switch()
    compare_psnr_switch()
    extract('result')
    plot('result','result4.png')

if __name__ == '__main__':
    main()