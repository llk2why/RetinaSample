import numpy as np
PATCH_SIZE = 128
PATCH_STRIDE = 64

class Dataset:
    RAW_DIR = r'C:\data\dataset\Sandwich 0612 fullsize'
    CHOPPED_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Chopped'
    # MOSAIC_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic'
    MOSAIC_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Deep DM_SR'
    CHOPPED_DIR_TEST = r'C:\data\dataset\Sandwich 0612 fullsize Chopped Test'
    # MOSAIC_DIR_TEST = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Test'
    MOSAIC_DIR_TEST = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Test Deep DM_SR'

class YAML:
    CHOP_PATCH = r'.\yamls\chop.yaml'


def generate_sample_matrix(shape):
    save_name = 'SampleTemplate.npy'
    sample = np.random.randint(0,3,shape)
    np.save(save_name,sample)
    print('Generated sampling matrix:\n{} \nand saved to file "{}" '.format(sample,save_name))

def generate_bayer_sample_matrix(shape):
    save_name = 'BayerTemplate.npy'
    tile = np.array([[0,1],[1,2]])
    sample = np.tile(tile,np.array(shape)//2)
    np.save(save_name,sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample,save_name))

if __name__ == '__main__':
    # generate_sample_matrix((PATCH_SIZE,PATCH_SIZE))
    generate_bayer_sample_matrix(((PATCH_SIZE,PATCH_SIZE)))
    pass
# SAMPLE_MATRIX = np.array(np.load('SampleTemplate.npy'))
SAMPLE_MATRIX = np.array(np.load('BayerTemplate.npy'))
