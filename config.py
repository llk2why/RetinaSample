import os
import platform
import numpy as np

PATCH_SIZE = 128
PATCH_STRIDE = 64

plat = platform.system()
root = r'D:\data\image dataset' if plat == 'Windows' else '/home/lincoln/data'


class Dataset:
    RAW_DIR = os.path.join(root, r'Sandwich 0612 fullsize')
    ##[LABEL] train label dir & test label dir
    CHOPPED_DIR = os.path.join(root, r'Sandwich 0612 fullsize Chopped')
    CHOPPED_DIR_TEST = os.path.join(root, r'Sandwich 0612 fullsize Chopped Test')

    ##[INPUT] train Bayer input dir & test Bayer input dir
    MOSAIC_DIR = os.path.join(root, r'Sandwich 0612 fullsize Mosaic Deep DM_SR')
    MOSAIC_DIR_TEST = os.path.join(root, r'Sandwich 0612 fullsize Mosaic Test Deep DM_SR')
    RESULT = os.path.join(root, r'Sandwich 0612 fullsize Mosaic Reconstruct Deep DM_SR')

    ##[INPUT] train RYYB input dir & test RYYB input dir
    RYYB_MOSAIC_DIR = os.path.join(root, r'Sandwich 0612 fullsize RYYB Mosaic')
    RYYB_MOSAIC_DIR_TEST = os.path.join(root, r'Sandwich 0612 fullsize RYYB Mosaic TEST')
    RYYB_RESULT = os.path.join(root, r'Sandwich 0612 fullsize Mosaic Reconstruct RYYB')

    ##[INPUT] train Random input dir & test Random input dir
    Random_MOSAIC_DIR = os.path.join(root, r'Sandwich 0612 fullsize Random Mosaic')
    Random_MOSAIC_DIR_TEST = os.path.join(root, r'Sandwich 0612 fullsize Random Mosaic TEST')
    Random_RESULT = os.path.join(root, r'Sandwich 0612 fullsize Mosaic Reconstruct Random')

    ##[INPUT] train Arbitrary input dir & test Arbitrary input dir
    Arbitrary_MOSAIC_DIR = os.path.join(root, r'Sandwich 0612 fullsize Arbitrary Mosaic')
    Arbitrary_MOSAIC_DIR_TEST = os.path.join(root, r'Sandwich 0612 fullsize Arbitrary Mosaic TEST')
    Arbitrary_RESULT = os.path.join(root, r'Sandwich 0612 fullsize Mosaic Reconstruct Arbitrary')

    ##[INPUT] train RB_G input dir & test RB_G input dir
    RB_G_MOSAIC_DIR = os.path.join(root, r'Sandwich 0612 fullsize RB_G Mosaic')
    RB_G_MOSAIC_DIR_TEST = os.path.join(root, r'Sandwich 0612 fullsize RB_G Mosaic TEST')
    RB_G_RESULT = os.path.join(root, r'Sandwich 0612 fullsize Mosaic Reconstruct RB_G')

    ##[INPUT] train RB_G input dir & test RB_G_DENOISE input dir
    RB_G_DENOISE_MOSAIC_DIR = RB_G_MOSAIC_DIR
    RB_G_DENOISE_MOSAIC_DIR_TEST = RB_G_MOSAIC_DIR_TEST
    RB_G_DENOISE_RESULT = os.path.join(root, r'Sandwich 0612 fullsize Mosaic Reconstruct RB_G_DENOISE')


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
    sample = np.random.randint(0, 3, shape)
    np.save(save_name, sample)
    print('Generated sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))


# BGGR sample matrix
def generate_bayer_sample_matrix(shape):
    save_name = 'BayerTemplate.npy'
    tile = np.array([[2, 1], [1, 0]])
    sample = np.tile(tile, np.array(shape) // 2)
    np.save(save_name, sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))


# RYYB sample matrix
def generate_ryyb_sample_matrix(shape):
    r, c = shape
    save_name = 'RYYBTemplate.npy'
    tile = np.array([[[1, 0, 0], [1, 1, 0]], [[1, 1, 0], [0, 0, 1]]])
    sample = np.tile(tile, (r // 2, c // 2, 1))
    np.save(save_name, sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))


# RB_G sample matrix
def generate_rb_g_sample_matrix(shape):
    r, c = shape
    save_name = 'RB_GTemplate.npy'
    tile = np.array([[[1, 1, 0], [0, 1, 1]], [[0, 1, 1], [1, 1, 0]]])
    sample = np.tile(tile, (r // 2, c // 2, 1))
    np.save(save_name, sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))


def generate_random_sample_matrix(shape):
    r, c = shape
    save_name = 'RandomTemplate.npy'
    tile = np.random.randint(0, 3, (r // 2, c // 2))
    cfa = np.tile(tile, (2, 2))
    sample = np.zeros((r, c, 3))
    for i in range(3):
        channel = sample[:, :, i]
        channel[cfa == i] = 1
    np.save(save_name, sample)
    print('Generated random sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))


# def generate_arbitrary_sample_matrix(shape):
#     r,c = shape
#     save_name = 'ArbitraryTemplate.npy'
#     tile = np.arbitrary.randint(0,3,(r//2,c//2))
#     cfa = np.tile(tile,(2,2))
#     sample = np.zeros((r,c,3))
#     for i in range(3):
#         channel = sample[:,:,i]
#         channel[cfa==i] = 1
#     np.save(save_name,sample)
#     print('Generated arbitrary sampling matrix:\n{} \nand saved to file "{}" '.format(sample,save_name))


if __name__ == '__main__':
    shape = (PATCH_SIZE, PATCH_SIZE)
    # generate_bayer_sample_matrix(shape)
    # generate_ryyb_sample_matrix(shape)
    # generate_random_sample_matrix(shape)
    generate_rb_g_sample_matrix(shape)

SAMPLE_MATRIX = np.array(np.load('BayerTemplate.npy'))
RYYB_SAMPLE_MATRIX = np.array(np.load('RYYBTemplate.npy'))
RANDOM_SAMPLE_MATRIX = np.array(np.load('RandomTemplate.npy'))
RB_G_SAMPLE_MATRIX = np.array(np.load('RB_GTemplate.npy'))
