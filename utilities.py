import os
import numpy as np

def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)

def load_train_data(index):
    fx = 'data/train_x{}.npy'.format(index)
    fy = 'data/train_y{}.npy'.format(index)
    train_x = np.load(fx)
    train_y = np.load(fy)
    return train_x,train_y

def load_test_data():
    test_x = np.load('data/test_x2.npy')
    test_y = np.load('data/test_y2.npy')
    test_x = np.load(fx)
    test_y = np.load(fy)
    return test_x,test_y

def read_train_name(index):
    with open('data/train_name{}.yaml'.format(index),'r') as f:
        train_name = yaml.load(f)
    return train_name

def read_test_name():
    with open('data/test_name.yaml','r') as f:
        test_name = yaml.load(f)
    return test_name