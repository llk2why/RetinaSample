import os
import platform
import numpy as np

PATCH_SIZE = 128
PATCH_STRIDE = 64

plat = platform.system()
root = r'C:\fastdata' if plat == 'Windows' else '/fastdata/retina'


def join_root(dirname):
    return os.path.join(root, dirname)


MOSAIC_DIR_TEMPLATE = join_root('Sandwich 0612 fullsize Mosaic {}')
MOSAIC_DIR_TEST_TEMPLATE = join_root('Sandwich 0612 fullsize Mosaic Test {}')
RESULT_TEMPLATE = join_root('Sandwich 0612 fullsize Mosaic Reconstruct {}')


class Dataset:
    RAW_DIR = join_root(r'Sandwich 0612 fullsize')

    # [LABEL] train label dir & test label dir
    CHOPPED_DIR = join_root(r'Sandwich 0612 fullsize Chopped')
    CHOPPED_DIR_TEST = join_root(r'Sandwich 0612 fullsize Chopped Test')

    # [INPUT] train Bayer input dir & test Bayer input dir
    Name = 'Deep DM_SR'
    MOSAIC_DIR = MOSAIC_DIR_TEMPLATE.format(Name)
    MOSAIC_DIR_TEST = MOSAIC_DIR_TEST_TEMPLATE.format(Name)
    RESULT = RESULT_TEMPLATE.format(Name)

    # [INPUT] train RYYB input dir & test RYYB input dir
    Name = 'RYYB'
    RYYB_MOSAIC_DIR = MOSAIC_DIR_TEMPLATE.format(Name)
    RYYB_MOSAIC_DIR_TEST = MOSAIC_DIR_TEST_TEMPLATE.format(Name)
    RYYB_RESULT = RESULT_TEMPLATE.format(Name)

    # [INPUT] train Random input dir & test Random input dir
    Name = 'Random'
    Random_MOSAIC_DIR = MOSAIC_DIR_TEMPLATE.format(Name)
    Random_MOSAIC_DIR_TEST = MOSAIC_DIR_TEST_TEMPLATE.format(Name)
    Random_RESULT = RESULT_TEMPLATE.format(Name)

    # [INPUT] train Arbitrary input dir & test Arbitrary input dir
    Name = 'Arbitrary'
    Arbitrary_MOSAIC_DIR = MOSAIC_DIR_TEMPLATE.format(Name)
    Arbitrary_MOSAIC_DIR_TEST = MOSAIC_DIR_TEST_TEMPLATE.format(Name)
    Arbitrary_RESULT = RESULT_TEMPLATE.format(Name)

    # [INPUT] train RB_G input dir & test RB_G input dir
    Name = 'RB_G'
    RB_G_MOSAIC_DIR = MOSAIC_DIR_TEMPLATE.format(Name)
    RB_G_MOSAIC_DIR_TEST = MOSAIC_DIR_TEST_TEMPLATE.format(Name)
    RB_G_RESULT = RESULT_TEMPLATE.format(Name)

    # [INPUT] train RB_G input dir & test RB_G_DENOISE input dir
    Name = 'RB_G_DENOISE'
    RB_G_DENOISE_MOSAIC_DIR = RB_G_MOSAIC_DIR
    RB_G_DENOISE_MOSAIC_DIR_TEST = RB_G_MOSAIC_DIR_TEST
    RB_G_DENOISE_RESULT = RESULT_TEMPLATE.format(Name)

    # [INPUT] train JointPixel_RGBG input dir & test JointPixel_RGBG input dir
    Name = 'JointPixel_RGBG'
    JointPixel_RGBG_MOSAIC_DIR = MOSAIC_DIR_TEMPLATE.format(Name)
    JointPixel_RGBG_MOSAIC_DIR_TEST = MOSAIC_DIR_TEST_TEMPLATE.format(Name)
    JointPixel_RGBG_RESULT = RESULT_TEMPLATE.format(Name)

    # [INPUT] train JointPixel_Triple input dir & test JointPixel_Triple input dir
    Name = 'JointPixel_Triple'
    JointPixel_Triple_MOSAIC_DIR = MOSAIC_DIR_TEMPLATE.format(Name)
    JointPixel_Triple_MOSAIC_DIR_TEST = MOSAIC_DIR_TEST_TEMPLATE.format(Name)
    JointPixel_Triple_RESULT = RESULT_TEMPLATE.format(Name)

    # [INPUT] train JointPixel_SR input dir & test JointPixel_SR input dir
    Name = 'JointPixel_SR'
    JointPixel_SR_MOSAIC_DIR = MOSAIC_DIR_TEMPLATE.format(Name)
    JointPixel_SR_MOSAIC_DIR_TEST = MOSAIC_DIR_TEST_TEMPLATE.format(Name)
    JointPixel_SR_RESULT = RESULT_TEMPLATE.format(Name)

    # [INPUT] train Paramized_RYYB input dir & test Paramized_RYYB input dir
    Name = 'Paramized_RYYB'
    Paramized_RYYB_MOSAIC_DIR = RYYB_MOSAIC_DIR
    Paramized_RYYB_MOSAIC_DIR_TEST = RYYB_MOSAIC_DIR_TEST
    Paramized_RYYB_RESULT = RESULT_TEMPLATE.format(Name)


""" 
    RAW_DIR = r'C:\data\dataset\Sandwich 0612 fullsize'
    CHOPPED_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Chopped'
    # MOSAIC_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic'
    MOSAIC_DIR = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Deep DM_SR'
    CHOPPED_DIR_TEST = r'C:\data\dataset\Sandwich 0612 fullsize Chopped Test'
    # MOSAIC_DIR_TEST = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Test'
    MOSAIC_DIR_TEST = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Test Deep DM_SR'
    RESULT = r'C:\data\dataset\Sandwich 0612 fullsize Mosaic Reconstruct Deep DM_SR' 
"""


class YAML:
    CHOP_PATCH = r'.\yamls\chop.yaml'


def generate_sample_matrix(shape):
    save_name = 'Template/SampleTemplate.npy'
    sample = np.random.randint(0, 3, shape)
    np.save(save_name, sample)
    print('Generated sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))


# BGGR sample matrix
def generate_bayer_sample_matrix(shape):
    save_name = 'Template/BayerTemplate.npy'
    tile = np.array([[2, 1], [1, 0]])
    sample = np.tile(tile, np.array(shape) // 2)
    np.save(save_name, sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))


# RYYB sample matrix
def generate_ryyb_sample_matrix(shape):
    r, c = shape
    save_name = 'Template/RYYBTemplate.npy'
    tile = np.array([[[1, 0, 0], [1, 1, 0]], [[1, 1, 0], [0, 0, 1]]])
    sample = np.tile(tile, (r // 2, c // 2, 1))
    np.save(save_name, sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))


# RB_G sample matrix
def generate_rb_g_sample_matrix(shape):
    r, c = shape
    save_name = 'Template/RB_GTemplate.npy'
    tile = np.array([[[1, 1, 0], [0, 1, 1]], [[0, 1, 1], [1, 1, 0]]])
    sample = np.tile(tile, (r // 2, c // 2, 1))
    np.save(save_name, sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))


def generate_random_sample_matrix(shape):
    r, c = shape
    save_name = 'Template/RandomTemplate.npy'
    tile = np.random.randint(0, 3, (r // 2, c // 2))
    cfa = np.tile(tile, (2, 2))
    sample = np.zeros((r, c, 3))
    for i in range(3):
        channel = sample[:, :, i]
        channel[cfa == i] = 1
    np.save(save_name, sample)
    print('Generated random sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))


def generate_jointpixel_rgbg_sample_matrix(shape):
    r, c = shape
    save_name = 'Template/JointPixel_RGBGTemplate.npy'
    tile = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
    tile = np.repeat(tile, 2, axis=0)
    sample = np.tile(tile, (r // 8, c // 4))
    np.save(save_name, sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))

def generate_jointpixel_triple_sample_matrix(shape):
    r, c = shape
    save_name = 'Template/JointPixel_TripleTemplate.npy'
    tile = np.array([[0,0,1,1], [1,0,1,2], [1,1,2,2]])
    pool = np.tile(tile, (r // 3 +1, c // 4 +1))
    sample = pool[:r,:c]
    np.save(save_name, sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))
    print(sample.shape)

def generate_jointpixel_sr_sample_matrix(shape):
    save_name = 'Template/JointPixel_SRTemplate.npy'
    tile = np.array([[0, 1], [1, 2]])
    tile = np.repeat(tile,2,axis=0)
    tile = np.repeat(tile,2,axis=1)
    sample = np.tile(tile, np.array(shape) // 4)
    np.save(save_name, sample)
    print('Generated bayer sampling matrix:\n{} \nand saved to file "{}" '.format(sample, save_name))
    print(sample.shape)


if __name__ == '__main__':
    shape = (PATCH_SIZE, PATCH_SIZE)
    # generate_bayer_sample_matrix(shape)
    # generate_ryyb_sample_matrix(shape)
    # generate_random_sample_matrix(shape)
    # generate_rb_g_sample_matrix(shape)
    # generate_jointpixel_rgbg_sample_matrix(shape)
    # generate_jointpixel_triple_sample_matrix(shape)
    generate_jointpixel_sr_sample_matrix(shape)
    

file_path = os.path.split(os.path.abspath(__file__))[0]

SAMPLE_MATRIX = np.array(np.load(os.path.join(file_path,'Template/BayerTemplate.npy')))
RYYB_SAMPLE_MATRIX = np.array(np.load(os.path.join(file_path,'Template/RYYBTemplate.npy')))
RANDOM_SAMPLE_MATRIX = np.array(np.load(os.path.join(file_path,'Template/RandomTemplate.npy')))
RB_G_SAMPLE_MATRIX = np.array(np.load(os.path.join(file_path,'Template/RB_GTemplate.npy')))
JOINTPIXEL_RGBG_MATRIX = np.array(np.load(os.path.join(file_path,'Template/JointPixel_RGBGTemplate.npy')))
JOINTPIXEL_Triple_MATRIX = np.array(np.load(os.path.join(file_path,'Template/JointPixel_TripleTemplate.npy')))
JOINTPIXEL_SR_MATRIX = np.array(np.load(os.path.join(file_path,'Template/JointPixel_SRTemplate.npy')))
