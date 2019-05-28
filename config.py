import numpy as np
PATCH_SIZE = 256
PATCH_STRIDE = 128

class Dataset:
    RAW_DIR = r'C:\data\dataset\Sandwich 0612 fullsize'
    CHOPPED_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Chopped'
    MOSAIC_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic'

class YAML:
    CHOP_PATCH = r'.\yamls\chop.yaml'


def generate_sample_matrix(shape):
    save_name = 'SampleTemplate.npy'
    sample = np.random.randint(0,3,shape)
    np.save(save_name,sample)
    print('Generated sampling matrix:\n{} \nand saved to file "{}" '.format(sample,save_name))

# generate_sample_matrix((PATCH_SIZE,PATCH_SIZE))

SAMPLE_MATRIX = np.array(np.load('SampleTemplate.npy'))
