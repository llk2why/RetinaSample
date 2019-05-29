import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import model
from utilities import LOG_INFO
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


parser = argparse.ArgumentParser()
parser.add_argument("--n_labels", default=4, type=int, help="Number of labels.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epoch.")
parser.add_argument("--batch_size", default=20, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=5, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.005, type=float, help="Learning rate for optimizer")
parser.add_argument("--debug", default=False, type=bool, help="Switch on debug")
parser.add_argument("--model_type", default='Decoder', type=str,
                    choices=['Decoder'], help="Available models")
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'
model = model.__dict__[args.model_type]().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()


def train(epoch, model, train_loader, optimizer, criterion):
    loss_list = []
    acc_list = []
    train_loss = []
    train_acc = []

    model.train()
    i = 1
    for batch,y in train_loader:
        batch,y = batch.to(device),y.to(device)
        optimizer.zero_grad()
        predictions = model(batch.float())
        loss = criterion(predictions, y.long())
        loss.backward()
        optimizer.step()

        acc = (predictions.max(1)[1] == y.long()).float().mean()
        loss_list.append(loss.item())
        acc_list.append(acc.item())

        train_loss.extend(loss_list)
        train_acc.extend(acc_list)

        if i % args.display_freq == 0:
            msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f, train acc = %.4f" % (
                epoch, i, len(train_loader), np.mean(loss_list), np.mean(acc_list)
            )
            LOG_INFO(msg)
            loss_list.clear()
            acc_list.clear()
        i+=1
    train_loss = np.mean(loss_list)
    train_acc = np.mean(acc_list)
    
    return train_loss,train_acc

def evaluate(model, criterion):
    model.eval()
    with torch.no_grad():
        tdata = test_data.to(device)
        predictions = model(tdata.float()).cpu()
    index = []
    cnt=-1
    ch = ''
    for i in range(test_id.size):
        cnt += int(ch!=test_id[i])
        ch = test_id[i]
        index.append(cnt)
    index = torch.tensor(index)
    pre = torch.zeros(np.unique(test_id).size,args.n_labels)
    pre.index_add_(0,index,predictions)
    pre = pre.max(1)[1]
    return pre+1

for epoch in range(1, args.epochs + 1):
    train_loss,train_acc = train(epoch, model, train_loader, optimizer, criterion)
    train_loss,train_acc = train(epoch, model, train_loader, optimizer, criterion)

