from Net_8_71 import Net
from Dataset_8_73 import Dataset
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
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
バッチサイズ1の1エポックの学習時間：2.1916234493255615sec
バッチサイズ2の1エポックの学習時間：1.2234456539154053sec
バッチサイズ4の1エポックの学習時間：0.6384384632110596sec
バッチサイズ8の1エポックの学習時間：0.3419332504272461sec
バッチサイズ16の1エポックの学習時間：0.1956191062927246sec
バッチサイズ32の1エポックの学習時間：0.11866879463195801sec
バッチサイズ64の1エポックの学習時間：0.08290886878967285sec
バッチサイズ128の1エポックの学習時間：0.06756925582885742sec
バッチサイズ256の1エポックの学習時間：0.0610051155090332sec
バッチサイズ512の1エポックの学習時間：0.05845952033996582sec
バッチサイズ1024の1エポックの学習時間：0.06722784042358398sec
"""