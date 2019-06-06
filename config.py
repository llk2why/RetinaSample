import os
import platform
import numpy as np
PATCH_SIZE = 128
PATCH_STRIDE = 64

plat = platform.system()
root = r'C:\data\dataset' if plat == 'Windows' else '/home/lincoln/data'
    
class Dataset:
    RAW_DIR = os.path.join(root,r'Sandwich 0612 fullsize')
    ##[LABEL] train label dir & test label dir
    CHOPPED_DIR = os.path.join(root,r'Sandwich 0612 fullsize Chopped')
    CHOPPED_DIR_TEST = os.path.join(root,r'Sandwich 0612 fullsize Chopped Test')

    ##[INPUT] train Bayer input dir & test Bayer input dir
    MOSAIC_DIR = os.path.join(root,r'Sandwich 0612 fullsize Mosaic Deep DM_SR')
    MOSAIC_DIR_TEST = os.path.join(root,r'Sandwich 0612 fullsize Mosaic Test Deep DM_SR')
    RESULT = os.path.join(root,r'Sandwich 0612 fullsize Mosaic Reconstruct Deep DM_SR')

    ##[INPUT] train RYYB input dir & test RYYB input dir
    RYYB_MOSAIC_DIR = os.path.join(root,r'Sandwich 0612 fullsize RYYB Mosaic')
    RYYB_MOSAIC_DIR_TEST = os.path.join(root,r'Sandwich 0612 fullsize RYYB Mosaic TEST')
    RYYB_RESULT = os.path.join(root,r'Sandwich 0612 fullsize Mosaic Reconstruct RYYB')

"""     RAW_DIR = r'C:\data\dataset\Sandwich 0612 fullsize'
    CHOPPED_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Chopped'
    # MOSAIC_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic'
    MOSAIC_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Deep DM_SR'
    CHOPPED_DIR_TEST = r'C:\data\dataset\Sandwich 0612 fullsize Chopped Test'
    # MOSAIC_DIR_TEST = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Test'
    MOSAIC_DIR_TEST = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Test Deep DM_SR'
    RESULT = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Reconstruct Deep DM_SR' """

class YAML:
    CHOP_PATCH = r'.\yamls\chop.yaml'


def generate_sample_matrix(shape):
    save_name = 'SampleTemplate.npy'
    sample = np.random.randint(0,3,shape)
    np.save(save_name,sample)
    print('Generated sampling matrix:\n{} \nand saved to file "{}" '.format(sample,save_name))

# BGGR sample matrix
def generate_bayer_sample_matrix(shape):
    save_name = 'BayerTemplate.npy'
    tile = np.array([[2,1],[1,0]])
    sample = np.tile(tile,np.array(shape)//2)
    np.save(save_name,sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample,save_name))

# RYYB sample matrix
def generate_ryyb_sample_matrix(shape):
    r,c = shape
    save_name = 'RYYBTemplate.npy'
    tile = np.array([[[1,0,0],[1,1,0]],[[1,1,0],[0,0,1]]])
    sample = np.tile(tile,(r//2,c//2,1))
    np.save(save_name,sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample,save_name))

if __name__ == '__main__':
    generate_bayer_sample_matrix(((PATCH_SIZE,PATCH_SIZE)))
    generate_ryyb_sample_matrix(((PATCH_SIZE,PATCH_SIZE)))

SAMPLE_MATRIX = np.array(np.load('BayerTemplate.npy'))
RYYB_SAMPLE_MATRIX = np.array(np.load('RYYBTemplate.npy'))
