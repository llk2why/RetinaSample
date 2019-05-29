import os
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
parser.add_argument("--epochs", default=50, type=int, help="Number of epoch.")
parser.add_argument("--batch_size", default=80, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=50, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--debug", default=False, type=bool, help="Switch on debug")
parser.add_argument("--model_type", default='Decoder', type=str,
                    choices=['Decoder'], help="Available models")
parser.add_argument("--threads", default=5, type=int, help="Worker number")
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'
model = model.__dict__[args.model_type]().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

train_data = DatasetFromFolder(Dataset.CHOPPED_DIR,Dataset.MOSAIC_DIR,
                               input_transform=Compose([ToTensor()]),
                               target_transform=Compose([ToTensor()]))
test_data = DatasetFromFolder(Dataset.CHOPPED_DIR_TEST,Dataset.MOSAIC_DIR_TEST,
                               input_transform=Compose([ToTensor()]),
                               target_transform=Compose([ToTensor()]))
train_loader = DataLoader(dataset=train_data, 
                          num_workers=args.threads, 
                          batch_size=args.batch_size, 
                          shuffle=True)
                          
test_loader = DataLoader(dataset=test_data,
                         num_workers=args.threads, 
                         batch_size=args.batch_size, 
                         shuffle=True)

def train(epoch, model, train_loader, optimizer, criterion):
    loss_list = []
    train_loss = []

    model.train()
    i = 1
    for x,y in train_loader:
        x,y = x.to(device),y.to(device)
        optimizer.zero_grad()
        predictions = model(x.float())
        loss = criterion(predictions, y.float())
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        mse = torch.sum(torch.pow(y-predictions,2),dim=[1,2,3])
        tensor = torch.tensor(255).float().to(device)
        psnr = 20*torch.log10(tensor)-10*torch.log10(mse)
        train_loss.extend(loss_list)

        if i % args.display_freq == 0:
            msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f, PSNR = %.4f" % (
                epoch, i, len(train_loader), np.mean(loss_list),np.mean(torch.mean(psnr).item())
            )
            LOG_INFO(msg)
            loss_list.clear()
        i+=1
    train_loss = np.mean(loss_list)
    
    return train_loss

def evaluate(model, criterion,loader):
    model.eval()
    i = 1
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            predictions = model(x.float())
            loss = criterion(predictions, y.float())
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            mse = torch.sum(torch.pow(y-predictions,2),dim=[1,2,3])
            tensor = torch.tensor(255).float().to(device)
            psnr = 20*torch.log10(tensor)-10*torch.log10(mse)
            train_loss.extend(loss_list)

            if i % args.display_freq == 0:
                msg = "[TEST]Epoch %02d, Iter [%03d/%03d], test loss = %.4f, PSNR = %.4f" % (
                    epoch, i, len(train_loader), np.mean(loss_list),np.mean(torch.mean(psnr).item())
                )
                LOG_INFO(msg)
                loss_list.clear()
            i+=1
        train_loss = np.mean(loss_list)
    
    return train_loss

if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, model, train_loader, optimizer, criterion)
        test_loss = evaluate(epoch, model, test_loader)
