import os
import cv2
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
import model

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from utilities import LOG_INFO, load_train_data, load_test_data

from config import *
from dataset import DatasetFromFolder
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from selflogger import SelfLogger

logger = SelfLogger('./log/log','run')

parser = argparse.ArgumentParser()
parser.add_argument("--n_labels", default=4, type=int, help="Number of labels.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epoch.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=20, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate for optimizer")
parser.add_argument("--debug", default=False, action='store_true', help="Switch on debug")
parser.add_argument("--noise", default=0, type=float, help="noise std")
parser.add_argument("--model_type", default='RB_G_DENOISE', type=str,
                    choices=['DemosaicSR', 'RYYB', 'Random', 'Arbitrary', 'RB_G', 'RB_G_DENOISE',\
                             'JointPixel_RGBG','JointPixel_Triple'],
                    help="Available models")
parser.add_argument("--threads", default=5, type=int, help="Worker number")
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'


def train(epoch, model, train_loader, optimizer, criterion):
    loss_list = []
    train_loss = []

    model.train()
    i = 1
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        predictions = model(x.float())
        loss = criterion(predictions, y.float())
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        mse = torch.mean(torch.pow(y - predictions, 2), dim=[1, 2, 3])
        tensor = torch.tensor(1).double().to(device)
        psnr = 20 * torch.log10(tensor) - 10 * torch.log10(mse)
        train_loss.extend(loss_list)
        if i % args.display_freq == 0:
            msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.8f, PSNR = %.4f" % (
                epoch, i, len(train_loader), np.mean(loss_list), np.mean(torch.mean(psnr).item())
            )
            LOG_INFO(msg)
            loss_list.clear()
        i += 1
    train_loss = np.mean(loss_list)

    return train_loss


def evaluate(epoch, model, loader, criterion, save=False, names=None):
    model.eval()
    loss_list = []
    test_loss = []
    i = 1
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predictions = model(x.float())
            loss = criterion(predictions, y.float())

            loss_list.append(loss.item())

            mse = torch.mean(torch.pow(y - predictions, 2), dim=[1, 2, 3])
            tensor = torch.tensor(1).float().to(device)
            psnr = 20 * torch.log10(tensor) - 10 * torch.log10(mse)
            test_loss.extend(loss_list)

            if i % args.display_freq == 0:
                msg = "[TEST]Epoch %02d, Iter [%03d/%03d], test loss = %.8f, PSNR = %.4f" % (
                    epoch, i, len(loader), np.mean(loss_list), np.mean(torch.mean(psnr).item())
                )
                LOG_INFO(msg)
                loss_list.clear()

            if save:
                # print('Saving results..')
                if args.model_type == 'DemosaicSR':
                    result_dir = Dataset.RESULT
                elif args.model_type == 'RYYB':
                    result_dir = Dataset.RYYB_RESULT
                elif args.model_type == 'Random':
                    result_dir = Dataset.Random_RESULT
                elif args.model_type == 'Arbitrary':
                    result_dir = Dataset.Arbitrary_RESULT
                elif args.model_type == 'RB_G':
                    result_dir = Dataset.RB_G_RESULT
                elif args.model_type == 'RB_G_DENOISE':
                    result_dir = Dataset.RB_G_DENOISE_RESULT
                elif args.model_type == 'JointPixel_RGBG':
                    result_dir = Dataset.JointPixel_RGBG_RESULT
                elif args.model_type == 'JointPixel_TripleTemplate':
                    result_dir = Dataset.JointPixel_Triple_RESULT
                if args.noise > 0:
                    result_dir = result_dir + ' noise={:.2f}'.format(args.noise)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                predictions = (predictions.cpu().numpy()).transpose(0, 2, 3, 1)
                n = predictions.shape[0]
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                for j in range(n):
                    fpath = os.path.join(result_dir, names[j + args.batch_size * (i - 1)])
                    img = (predictions[j] * 255)
                    img[img > 255] = 255
                    img[img < 0] = 0
                    img = img.astype(np.uint8)
                    cv2.imwrite(fpath, img)
            i += 1
        test_loss = np.mean(loss_list)

    return test_loss


def main():
    start_time = time.time()
    LOG_INFO('===> Building model')
    net = model.__dict__[args.model_type](4).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.MSELoss()

    if args.model_type == 'DemosaicSR':
        train_x_dir = Dataset.MOSAIC_DIR
        test_x_dir = Dataset.MOSAIC_DIR_TEST
    elif args.model_type == 'RYYB':
        train_x_dir = Dataset.RYYB_MOSAIC_DIR
        test_x_dir = Dataset.RYYB_MOSAIC_DIR_TEST
    elif args.model_type == 'Random':
        train_x_dir = Dataset.Random_MOSAIC_DIR
        test_x_dir = Dataset.Random_MOSAIC_DIR_TEST
    elif args.model_type == 'Arbitrary':
        train_x_dir = Dataset.Arbitrary_MOSAIC_DIR
        test_x_dir = Dataset.Arbitrary_MOSAIC_DIR_TEST
    elif args.model_type == 'RB_G':
        train_x_dir = Dataset.RB_G_MOSAIC_DIR
        test_x_dir = Dataset.RB_G_MOSAIC_DIR_TEST
    elif args.model_type == 'RB_G_DENOISE':
        train_x_dir = Dataset.RB_G_DENOISE_MOSAIC_DIR
        test_x_dir = Dataset.RB_G_DENOISE_MOSAIC_DIR_TEST
    elif args.model_type == 'JointPixel_RGBG':
        train_x_dir = Dataset.JointPixel_RGBG_MOSAIC_DIR
        test_x_dir = Dataset.JointPixel_RGBG_MOSAIC_DIR_TEST
    elif args.model_type == 'JointPixel_Triple':
        train_x_dir = Dataset.JointPixel_Triple_MOSAIC_DIR
        test_x_dir = Dataset.JointPixel_Triple_MOSAIC_DIR_TEST
    train_y_dir = Dataset.CHOPPED_DIR
    test_y_dir = Dataset.CHOPPED_DIR_TEST

    LOG_INFO('===> Loading datasets')
    train_data = DatasetFromFolder(train_x_dir, train_y_dir, args.model_type,
                                   input_transform=Compose([ToTensor()]),
                                   target_transform=Compose([ToTensor()]),
                                   debug=args.debug, noise=args.noise)
    test_data = DatasetFromFolder(test_x_dir, test_y_dir, args.model_type,
                                  input_transform=Compose([ToTensor()]),
                                  target_transform=Compose([ToTensor()]),
                                  debug=args.debug, noise=args.noise)
    train_loader = DataLoader(dataset=train_data,
                              num_workers=args.threads,
                              batch_size=args.batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_data,
                             num_workers=args.threads,
                             batch_size=args.batch_size,
                             shuffle=False)

    LOG_INFO('===> Begin training and testing')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, net, train_loader, optimizer, criterion)
        # test_loss = evaluate(epoch, net, test_loader, criterion)
    LOG_INFO('===> FINISH TRAINING')
    test_loss = evaluate(0, net, test_loader, criterion, save=True, names=test_data.filenames)
    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(net.state_dict(),
               'model/{}_model_noise={:.2f}_epoch={}.pth'.format(args.model_type, args.noise, args.epochs))
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info('Finish {} with noise {} in {:.2f} seconds.'.format(args.model_type,args.noise,duration))


if __name__ == '__main__':
    main()
