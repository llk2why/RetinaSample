import os
import cv2
import yaml
import numpy as np
from utilities import LOG_INFO

# LOG_INFO('====> Begin reading yaml')
# with open('./yamls/chop.yaml') as f:
#     PATCH_INFO = yaml.load(f)

names = [
        'SDIM1895',
        'SDIM1898',
        'SDIM1899',
        'SDIM1901',
        'SDIM1905',
        'SDIM1906',
        'SDIM1912',
        'SDIM1916',
        'SDIM1918',
        'SDIM1920',
        'SDIM1925',
        'SDIM1926',
        'SDIM1927',
        'SDIM1928',
        'SDIM1929',
        'SDIM1930',
        'SDIM1931',
        'SDIM1933',
    ]

name_pair = {
    's1895':'SDIM1895',
    's1898':'SDIM1898',
    's1899':'SDIM1899',
    's1901':'SDIM1901',
    's1905':'SDIM1905',
    's1906':'SDIM1906',
    's1912':'SDIM1912',
    's1916':'SDIM1916',
    's1918':'SDIM1918',
    's1920':'SDIM1920',
    's1925':'SDIM1925',
    's1926':'SDIM1926',
    's1927':'SDIM1927',
    's1928':'SDIM1928',
    's1929':'SDIM1929',
    's1930':'SDIM1930',
    's1931':'SDIM1931',
    's1933':'SDIM1933',
}

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
    LOG_INFO('====> Begin joint patches')
    for i,name in enumerate(names):
        LOG_INFO('====> {}/{}'.format(i+1,len(names)))
        joint_patch(patch_dir,name,tag)

def compare_psnr(img_dir,suffix,tag):
    files = [x for x in os.listdir(img_dir) if suffix in x]
    psnrs = []
    with open('psnr_{}.txt'.format(tag),'w') as f:
        for file in files:
            fpath = os.path.join(img_dir,file)
            if 's' in file:
                std_name = name_pair[os.path.splitext(file)[0].split('_')[0]]
            else:
                std_name = file.split('.')[0]
            std_fpath = os.path.join('C:\data\dataset\Sandwich 0612 fullsize',std_name+'.TIF')
            print(fpath)
            print(std_fpath)
            im_y = cv2.imread(fpath)
            im_x = cv2.imread(std_fpath)
            psnr = get_psnr(im_x,im_y)
            psnrs.append(psnr)
            f.write('{}:{}\n'.format(file,psnr))
        f.write('avg:{}\n'.format(np.mean(psnrs)))

def main():
    # RGGB_dir = r'C:\data\dataset\result best'
    # combine(RGGB_dir,'RGGB')
    # RYYB_dir = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Reconstruct RYYB'
    # combine(RYYB_dir,'RYYB')
    compare_psnr(r'C:\Users\linc\Downloads\L01\output','png','RGGB(CS)')
    compare_psnr(r'joint(RGGB)','tiff','RGGB')
    compare_psnr(r'joint(RYYB)','tiff','RYYB')
if __name__ == '__main__':
    main()
