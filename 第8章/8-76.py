from Net_8_71 import Net
from Dataset_8_73 import Dataset
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

with open('train.feature.pickle', 'rb') as f:
    train_vectors = pickle.load(f)
    
with open('train.label.pickle', 'rb') as f:
    train_labels = pickle.load(f)
    
with open('valid.feature.pickle', 'rb') as f:
    valid_vectors = pickle.load(f)
    
with open('valid.label.pickle', 'rb') as f:
    valid_labels = pickle.load(f)
    
train_dataset = Dataset(train_vectors, train_labels)
valid_dataset = Dataset(valid_vectors, valid_labels)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

model = Net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    for i, data in enumerate(train_dataloader):
        
        inputs, labels = data
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        torch.save(optimizer.state_dict(), f'alg_checkpoint{epoch+1}.pt')
        torch.save(model.state_dict(), f'param_checkpoint{epoch+1}.pt')