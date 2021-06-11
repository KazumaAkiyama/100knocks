from Net_8_71 import Net
from Dataset_8_73 import Dataset
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

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

class MLNet(nn.Module):
    def __init__(self):
        super(MLNet, self).__init__()
        self.fc1 = nn.Linear(300, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = MLNet()

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

for epoch in range(10):
    
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        
        inputs, labels = data
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

train_loss, train_accuracy = loss_and_accuracy(model, train_dataloader, criterion)
test_loss, test_accuracy = loss_and_accuracy(model, test_dataloader, criterion)
    
print(f"学習データの損失 : {train_loss}")
print(f"学習データの正解率 : {train_accuracy}")
print(f"評価データの損失 : {test_loss}")
print(f"評価データの正解率 : {test_accuracy}")

"""
[1,  2000] loss: 1.062
[1,  4000] loss: 0.597
[1,  6000] loss: 0.524
[1,  8000] loss: 0.479
[1, 10000] loss: 0.429
[2,  2000] loss: 0.332
[2,  4000] loss: 0.333
[2,  6000] loss: 0.334
[2,  8000] loss: 0.305
[2, 10000] loss: 0.294
[3,  2000] loss: 0.236
[3,  4000] loss: 0.301
[3,  6000] loss: 0.287
[3,  8000] loss: 0.267
[3, 10000] loss: 0.261
[4,  2000] loss: 0.238
[4,  4000] loss: 0.243
[4,  6000] loss: 0.242
[4,  8000] loss: 0.274
[4, 10000] loss: 0.239
[5,  2000] loss: 0.240
[5,  4000] loss: 0.227
[5,  6000] loss: 0.226
[5,  8000] loss: 0.231
[5, 10000] loss: 0.232
[6,  2000] loss: 0.198
[6,  4000] loss: 0.220
[6,  6000] loss: 0.231
[6,  8000] loss: 0.231
[6, 10000] loss: 0.231
[7,  2000] loss: 0.204
[7,  4000] loss: 0.205
[7,  6000] loss: 0.238
[7,  8000] loss: 0.209
[7, 10000] loss: 0.221
[8,  2000] loss: 0.177
[8,  4000] loss: 0.202
[8,  6000] loss: 0.214
[8,  8000] loss: 0.193
[8, 10000] loss: 0.214
[9,  2000] loss: 0.184
[9,  4000] loss: 0.189
[9,  6000] loss: 0.180
[9,  8000] loss: 0.200
[9, 10000] loss: 0.194
[10,  2000] loss: 0.157
[10,  4000] loss: 0.178
[10,  6000] loss: 0.190
[10,  8000] loss: 0.192
[10, 10000] loss: 0.152
Finished Training
学習データの損失 : 0.1578063245942388
学習データの正解率 : 0.9448708348932984
評価データの損失 : 0.279944483614518
評価データの正解率 : 0.9109281437125748
"""