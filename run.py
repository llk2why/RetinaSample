import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import model

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from utilities import LOG_INFO,load_train_data,load_test_data

from config import *
from dataset import DatasetFromFolder
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


parser = argparse.ArgumentParser()
parser.add_argument("--n_labels", default=4, type=int, help="Number of labels.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epoch.")
parser.add_argument("--batch_size", default=100, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=50, type=int, help="Display frequency")
parser.add_argument("--lr", default=10, type=float, help="Learning rate for optimizer")
parser.add_argument("--debug", default=False, type=bool, help="Switch on debug")
parser.add_argument("--model_type", default='Decoder2', type=str,
                    choices=['Decoder,Decoder2'], help="Available models")
parser.add_argument("--threads", default=5, type=int, help="Worker number")
args = parser.parse_args()
use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'


def train(epoch, model, train_loader, optimizer, criterion):
    loss_list = []
    train_loss = []

    model = model.train()
    i = 1
    for x,y in train_loader:
        # print(x.shape,y.shape)
        x,y = x.to(device),y.to(device)
        predictions = model(x)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        # print(loss.grad)
        loss.backward()
        # print(loss.grad)
        optimizer.step()
        

        # x = (x[5].cpu().numpy()).astype(np.uint8).transpose(1,2,0)
        # cv2.imwrite('x.jpg',x)
        # y = (y[5].cpu().numpy()).astype(np.uint8).transpose(1,2,0)
        # cv2.imwrite('y.jpg',y)
        # pre = (predictions[5].clone().detach().cpu().numpy()).astype(np.uint8).transpose(1,2,0)
        # cv2.imwrite('z.jpg',pre)
        # print(np.max(x),np.max(y),np.max(pre))
        # exit()
        # print('max:')
        # print(torch.max(x).detach().cpu().numpy(),\
        #     torch.max(y).detach().cpu().numpy()\
        #     ,torch.max(predictions).detach().cpu().numpy())
        


        loss_list.append(loss.item())

        mse = torch.mean(torch.pow(y-predictions,2),dim=[1,2,3])
        tensor = torch.tensor(255).double().to(device)
        psnr = 20*torch.log10(tensor)-10*torch.log10(mse)
        train_loss.extend(loss_list)

        # print(torch.max(model.layers.c1.weight.grad).cpu().numpy(),end=' ')
        # print(torch.max(model.layers.c2.weight.grad).cpu().numpy(),end=' ')
        # print(torch.max(model.layers.c3.weight.grad).cpu().numpy())
        
        # y = predictions[-1].clone().detach().cpu().numpy()
        # print(y.shape)
        # cv2.imshow('y',y.transpose(2,0,1))
        # cv2.waitKey(0)

        if i % args.display_freq == 0:
            # exit()
            msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f, PSNR = %.4f" % (
                epoch, i, len(train_loader), np.mean(loss_list),np.mean(torch.mean(psnr).item())
            )
            LOG_INFO(msg)
            loss_list.clear()
        i+=1
    train_loss = np.mean(loss_list)
    
    return train_loss

def evaluate(epoch,model,loader,criterion,save=False,names=None):
    model = model.eval()
    loss_list = []
    test_loss = []
    i = 1
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            predictions = model(x)
            loss = criterion(predictions, y)

            loss_list.append(loss.item())

            mse = torch.mean(torch.pow(y-predictions,2),dim=[1,2,3])
            tensor = torch.tensor(255).double().to(device)
            psnr = 20*torch.log10(tensor)-10*torch.log10(mse)
            test_loss.extend(loss_list)

            if i % args.display_freq == 0:
                msg = "[TEST]Epoch %02d, Iter [%03d/%03d], test loss = %.4f, PSNR = %.4f" % (
                    epoch, i, len(loader), np.mean(loss_list),np.mean(torch.mean(psnr).item())
                )
                LOG_INFO(msg)
                loss_list.clear()
            if save:
                predictions = (predictions.cpu().numpy()).astype(np.uint8).transpose(0,2,3,1)
                n = predictions.shape[0]
                if not os.path.exists(Dataset.RESULT):
                    os.makedirs(Dataset.RESULT)
                for j in range(n):
                    fpath = os.path.join(Dataset.RESULT,names[j+args.batch_size*(i-1)])
                    img = predictions[j]
                    cv2.imwrite(fpath,img)
            i+=1
        test_loss = np.mean(loss_list)
        
    
    return test_loss

def main():
    print('===> Building model')
    net = model.__dict__[args.model_type]().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print('===> Loading datasets')
    train_data = DatasetFromFolder(Dataset.MOSAIC_DIR,Dataset.CHOPPED_DIR,
                                # input_transform=Compose([ToTensor()]),
                                # target_transform=Compose([ToTensor()])
                                )
    test_data = DatasetFromFolder(Dataset.MOSAIC_DIR_TEST,Dataset.CHOPPED_DIR_TEST,
                                # input_transform=Compose([ToTensor()]),
                                # target_transform=Compose([ToTensor()])
                                )
    train_loader = DataLoader(dataset=train_data, 
                            num_workers=args.threads, 
                            batch_size=args.batch_size, 
                            shuffle=False)
                            
    test_loader = DataLoader(dataset=test_data,
                            num_workers=args.threads, 
                            batch_size=args.batch_size, 
                            shuffle=False)

    print('===> Begin training and testing')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, net, train_loader, optimizer, criterion)
        test_loss = evaluate(epoch, net, test_loader, criterion)
    test_loss = evaluate(0, net, test_loader, criterion,save=True,names=test_data.filenames)

if __name__ == '__main__':
    main()
