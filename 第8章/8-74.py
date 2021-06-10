from Net_8_71 import Net
from Dataset_8_73 import Dataset
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

with open('train.feature.pickle', 'rb') as f:
    train_vectors = pickle.load(f)
    
with open('train.label.pickle', 'rb') as f:
    train_labels = pickle.load(f)
    
with open('test.feature.pickle', 'rb') as f:
    test_vectors = pickle.load(f)
    
with open('test.label.pickle', 'rb') as f:
    test_labels = pickle.load(f)
    
train_dataset = Dataset(train_vectors, train_labels)
test_dataset = Dataset(test_vectors, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = Net()
model.load_state_dict(torch.load('model_weights.pth'))

def accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
    return correct / total

print(f"学習データの正解率 : {accuracy(model, train_dataloader)}")
print(f"評価データの正解率 : {accuracy(model, test_dataloader)}")

"""
学習データの正解率 : 0.8998502433545489
評価データの正解率 : 0.8944610778443114
"""