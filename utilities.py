import os
import numpy as np

def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)

def loadtrain1():
    train_x = np.load('train_x1.npy')
    train_y = np.load('train_y1.npy')
    with open('train_name1.yaml','r') as f:
        train_name = yaml.load(f)
    return train_x,train_y,train_name

def loadtrain2():
    train_x = np.load('train_x2.npy')
    train_y = np.load('train_y2.npy')
    with open('train_name2.yaml','r') as f:
        train_name = yaml.load(f)
    return train_x,train_y,train_name

def loadtest():
    test_x = np.load('test_x2.npy')
    test_y = np.load('test_y2.npy')
    with open('test_name.yaml','r') as f:
        test_name = yaml.load(f)
    return test_x,test_y,test_name