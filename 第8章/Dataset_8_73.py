from Net_8_71 import Net
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

with open('train.feature.pickle', 'rb') as f:
    train_vectors = pickle.load(f)
    
with open('train.label.pickle', 'rb') as f:
    train_labels = pickle.load(f)
    
with open('valid.feature.pickle', 'rb') as f:
    valid_vectors = pickle.load(f)
    
with open('valid.label.pickle', 'rb') as f:
    valid_labels = pickle.load(f)
    
with open('test.feature.pickle', 'rb') as f:
    test_vectors = pickle.load(f)
    
with open('test.label.pickle', 'rb') as f:
    test_labels = pickle.load(f)

class Dataset(Dataset):
    def __init__(self, x, t):
        self.x = x
        self.t = t
        self.size = len(x)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return [self.x[index], self.t[index]]
    
train_dataset = Dataset(train_vectors, train_labels)
valid_dataset = Dataset(valid_vectors, valid_labels)
test_dataset = Dataset(test_vectors, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = Net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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

torch.save(model.state_dict(), 'model_weights.pth')

"""
g03:~/第8章> python3 8-73.py
tensor([0.2321, 0.3031, 0.2300, 0.2348], grad_fn=<SoftmaxBackward>)
tensor([[0.2321, 0.3031, 0.2300, 0.2348],
        [0.2604, 0.2477, 0.2323, 0.2596],
        [0.2621, 0.2450, 0.2523, 0.2405],
        [0.2048, 0.2640, 0.2162, 0.3150]], grad_fn=<SoftmaxBackward>)
[1,  2000] loss: 0.960
[1,  4000] loss: 0.728
[1,  6000] loss: 0.654
[1,  8000] loss: 0.564
[1, 10000] loss: 0.563
[2,  2000] loss: 0.509
[2,  4000] loss: 0.506
[2,  6000] loss: 0.457
[2,  8000] loss: 0.442
[2, 10000] loss: 0.427
[3,  2000] loss: 0.421
[3,  4000] loss: 0.414
[3,  6000] loss: 0.377
[3,  8000] loss: 0.422
[3, 10000] loss: 0.399
[4,  2000] loss: 0.387
[4,  4000] loss: 0.385
[4,  6000] loss: 0.375
[4,  8000] loss: 0.352
[4, 10000] loss: 0.361
[5,  2000] loss: 0.370
[5,  4000] loss: 0.345
[5,  6000] loss: 0.358
[5,  8000] loss: 0.320
[5, 10000] loss: 0.367
[6,  2000] loss: 0.328
[6,  4000] loss: 0.346
[6,  6000] loss: 0.343
[6,  8000] loss: 0.339
[6, 10000] loss: 0.319
[7,  2000] loss: 0.324
[7,  4000] loss: 0.329
[7,  6000] loss: 0.328
[7,  8000] loss: 0.306
[7, 10000] loss: 0.330
[8,  2000] loss: 0.308
[8,  4000] loss: 0.317
[8,  6000] loss: 0.290
[8,  8000] loss: 0.322
[8, 10000] loss: 0.327
[9,  2000] loss: 0.333
[9,  4000] loss: 0.277
[9,  6000] loss: 0.302
[9,  8000] loss: 0.318
[9, 10000] loss: 0.313
[10,  2000] loss: 0.316
[10,  4000] loss: 0.287
[10,  6000] loss: 0.296
[10,  8000] loss: 0.312
[10, 10000] loss: 0.305
Finished Training
"""