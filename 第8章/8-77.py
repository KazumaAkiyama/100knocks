from Net_8_71 import Net
from Dataset_8_73 import Dataset
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time

with open('train.feature.pickle', 'rb') as f:
    train_vectors = pickle.load(f)
    
with open('train.label.pickle', 'rb') as f:
    train_labels = pickle.load(f)
    
train_dataset = Dataset(train_vectors, train_labels)

model = Net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def learning_time(train_dataset, model, criterion, optimizer, batch_size):
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    for i, data in enumerate(train_dataloader):

        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    
    return end_time - start_time

for i in range(11):
    batch_size = 2 ** i
    print(f"バッチサイズ{batch_size}の1エポックの学習時間：{learning_time(train_dataset, model, criterion, optimizer, batch_size)}sec")
    
    
"""
バッチサイズ1の1エポックの学習時間：2.338005542755127sec
バッチサイズ2の1エポックの学習時間：1.197274923324585sec
バッチサイズ4の1エポックの学習時間：0.6191196441650391sec
バッチサイズ8の1エポックの学習時間：0.32768750190734863sec
バッチサイズ16の1エポックの学習時間：0.18233084678649902sec
バッチサイズ32の1エポックの学習時間：0.10920095443725586sec
バッチサイズ64の1エポックの学習時間：0.07145190238952637sec
バッチサイズ128の1エポックの学習時間：0.09173393249511719sec
バッチサイズ256の1エポックの学習時間：0.04639458656311035sec
バッチサイズ512の1エポックの学習時間：0.0404050350189209sec
バッチサイズ1024の1エポックの学習時間：0.03793525695800781sec
"""