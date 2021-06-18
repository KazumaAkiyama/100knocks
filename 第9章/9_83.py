from RNN_9_81 import newDataset, Sampler, DescendingSampler, RNN, gen_loader, gen_descending_loader
from Train_9_82 import Task, Trainer, Predictor
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.optim as optim
import pickle
import numpy as np

with open('train.ids.pickle', 'rb') as f:
    train_ids = pickle.load(f)
    
with open('train.label.pickle', 'rb') as f:
    train_labels = pickle.load(f)

with open('valid.ids.pickle', 'rb') as f:
    valid_ids = pickle.load(f)
    
with open('valid.label.pickle', 'rb') as f:
    valid_labels = pickle.load(f)

with open('test.ids.pickle', 'rb') as f:
    test_ids = pickle.load(f)
    
with open('test.label.pickle', 'rb') as f:
    test_labels = pickle.load(f)

with open('word_dict.pickle', 'rb') as f:
    word_dict = pickle.load(f)

train_dataset = newDataset(train_ids, train_labels)
valid_dataset = newDataset(valid_ids, valid_labels)
test_dataset = newDataset(test_ids, test_labels)

#デバイスにcudaを指定，以降GPU上で学習が進む
device = torch.device('cuda')

model = RNN(len(word_dict), 300, 128, 4)

#バッチサイズを128に指定
loaders = (
    gen_descending_loader(train_dataset, 128),
    gen_descending_loader(valid_dataset, 128),
)
task = Task()
optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, nesterov=True)
trainer = Trainer(model, loaders, task, optimizer, 10, device)
trainer.train()

predictor = Predictor(model, gen_loader(train_dataset, 1), device)
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_dataset, pred))

predictor = Predictor(model, gen_loader(test_dataset, 1), device)
pred = predictor.predict()
print('評価データでの正解率 :', accuracy(test_dataset, pred))

"""
epoch 1, train_loss:1.1271446555445594, valid_loss:1.0133194062482105
epoch 2, train_loss:0.8148592196120384, valid_loss:0.7189750462491414
epoch 3, train_loss:0.5795994221848657, valid_loss:0.5663463799785194
学習データでの正解率 : 0.8208536128790715
評価データでの正解率 : 0.8008982035928144
"""