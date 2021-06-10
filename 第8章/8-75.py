from Net_8_71 import Net
from Dataset_8_73 import Dataset
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

def loss_and_accuracy(model, dataloader, criterion):
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(dataloader), correct / total

train_losses = []
train_accuracys = []
valid_losses = []
valid_accuracys = []

for epoch in range(10):
    
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        
        inputs, labels = data
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    train_loss, train_accuracy = loss_and_accuracy(model, train_dataloader, criterion)
    valid_loss, valid_accuracy = loss_and_accuracy(model, valid_dataloader, criterion)
    
    train_losses.append(train_loss)
    train_accuracys.append(train_accuracy)
    valid_losses.append(valid_loss)
    valid_accuracys.append(valid_accuracy)
    
fig, axes = plt.subplots(1,2)
x = np.arange(0, 10, 1)
y = np.array(train_losses)
axes[0].plot(x, y, label="train")
x = np.arange(0, 10, 1)
y = np.array(valid_losses)
axes[0].plot(x, y, label="valid")
axes[0].legend()
x = np.arange(0, 10, 1)
y = np.array(train_accuracys)
axes[1].plot(x, y, label="train")
x = np.arange(0, 10, 1)
y = np.array(valid_accuracys)
axes[1].plot(x, y, label="valid")
axes[1].legend()
fig.savefig("8-75.png")