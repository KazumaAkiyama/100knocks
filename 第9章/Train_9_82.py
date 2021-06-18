from RNN_9_81 import newDataset, Sampler, DescendingSampler, RNN, gen_loader, gen_descending_loader
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

#ロスの計算を行うクラス
class Task:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, model, batch):
        model.zero_grad()
        loss = self.criterion(model(batch), batch['labels'])
        loss.backward()
        return loss.item()

    def valid_step(self, model, batch):
        with torch.no_grad():
            loss = self.criterion(model(batch), batch['labels'])
        return loss.item()

#学習を回すクラス
class Trainer:
    def __init__(self, model, loaders, task, optimizer, max_iter, device = None):
        self.model = model
        self.model.to(device)
        self.train_loader, self.valid_loader = loaders
        self.task = task
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.device = device

    #データをいちいちdeviceに送る
    def send(self, batch):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch

    #学習データから学習を行う．平均ロスも計算する．
    def train_epoch(self):
        self.model.train()
        acc = 0
        for n, batch in enumerate(self.train_loader):
            batch = self.send(batch)
            acc += self.task.train_step(self.model, batch)
            self.optimizer.step()
        return acc / n

    #検証データから平均ロスを計算する
    def valid_epoch(self):
        self.model.eval()
        acc = 0
        for n, batch in enumerate(self.valid_loader):
            batch = self.send(batch)
            acc += self.task.valid_step(self.model, batch)
        return acc / n

    #学習を行うメソッド．エポックごとにロスを算出する
    def train(self):
        for epoch in range(self.max_iter):
            train_loss = self.train_epoch()
            valid_loss = self.valid_epoch()
            print(f'epoch {epoch+1}, train_loss:{train_loss}, valid_loss:{valid_loss}')

#予測用クラス
class Predictor:
    def __init__(self, model, loader, device=None):
        self.model = model
        self.loader = loader
        self.device = device

    #データをいちいちdeviceに送る
    def send(self, batch):
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch

    #予測結果を出す
    def infer(self, batch):
        self.model.eval()
        batch = self.send(batch)
        return self.model(batch).argmax(dim=-1).item()

    #予測確率のリストをだす
    def predict(self):
        lst = []
        for batch in self.loader:
            lst.append(self.infer(batch))
        return lst

#データセットの読み込み
train_dataset = newDataset(train_ids, train_labels)
valid_dataset = newDataset(valid_ids, valid_labels)
test_dataset = newDataset(test_ids, test_labels)

#モデルの生成
model = RNN(len(word_dict), 300, 50, 4)
loaders = (
    gen_loader(train_dataset, 1),
    gen_loader(valid_dataset, 1),
)

#タスククラスの生成
task = Task()

#オプティマイザーの生成
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#学習用クラスの生成
trainer = Trainer(model, loaders, task, optimizer, 3)
#学習の実行
trainer.train()

#正解率の計算をする
def accuracy(true, pred):
    return np.mean([t == p for t, p in zip(true, pred)])

#予測結果をaccuracy関数に投げて正解率を計算する
predictor = Predictor(model, gen_loader(train_dataset, 1))
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_labels, pred))

predictor = Predictor(model, gen_loader(test_dataset, 1))
pred = predictor.predict()
print('評価データでの正解率 :', accuracy(test_labels, pred))

"""
tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
epoch 1, train_loss:1.1213019774790998, valid_loss:0.958223113194387
epoch 2, train_loss:0.8034565940335062, valid_loss:0.7457776115656122
epoch 3, train_loss:0.5835454005571149, valid_loss:0.5658701492773459
学習データでの正解率 : 0.8205728191688506
評価データでの正解率 : 0.7941616766467066
"""