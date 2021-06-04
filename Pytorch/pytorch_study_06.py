import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

learning_rate = 1e-3
batch_size = 64
epochs = 5

learning_rate = 1e-3
batch_size = 64
epochs = 5

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

"""
g03:~/Pytorch> python3 pytorch_study_06.py
Epoch 1
-------------------------------
loss: 2.301850  [    0/60000]
loss: 2.301772  [ 6400/60000]
loss: 2.287602  [12800/60000]
loss: 2.282266  [19200/60000]
loss: 2.288321  [25600/60000]
loss: 2.266374  [32000/60000]
loss: 2.275126  [38400/60000]
loss: 2.260418  [44800/60000]
loss: 2.257571  [51200/60000]
loss: 2.218333  [57600/60000]
Test Error:
 Accuracy: 29.9%, Avg loss: 0.035088

Epoch 2
-------------------------------
loss: 2.262355  [    0/60000]
loss: 2.266269  [ 6400/60000]
loss: 2.227024  [12800/60000]
loss: 2.199205  [19200/60000]
loss: 2.222566  [25600/60000]
loss: 2.188939  [32000/60000]
loss: 2.182899  [38400/60000]
loss: 2.168985  [44800/60000]
loss: 2.172469  [51200/60000]
loss: 2.069358  [57600/60000]
Test Error:
 Accuracy: 36.4%, Avg loss: 0.033181

Epoch 3
-------------------------------
loss: 2.184097  [    0/60000]
loss: 2.180282  [ 6400/60000]
loss: 2.098815  [12800/60000]
loss: 2.044453  [19200/60000]
loss: 2.057592  [25600/60000]
loss: 2.029760  [32000/60000]
loss: 2.010231  [38400/60000]
loss: 1.991853  [44800/60000]
loss: 2.035822  [51200/60000]
loss: 1.820437  [57600/60000]
Test Error:
 Accuracy: 48.4%, Avg loss: 0.030049

Epoch 4
-------------------------------
loss: 2.052242  [    0/60000]
loss: 2.031134  [ 6400/60000]
loss: 1.897551  [12800/60000]
loss: 1.819788  [19200/60000]
loss: 1.815971  [25600/60000]
loss: 1.813320  [32000/60000]
loss: 1.783175  [38400/60000]
loss: 1.769439  [44800/60000]
loss: 1.877445  [51200/60000]
loss: 1.548434  [57600/60000]
Test Error:
 Accuracy: 50.3%, Avg loss: 0.026651

Epoch 5
-------------------------------
loss: 1.899816  [    0/60000]
loss: 1.861221  [ 6400/60000]
loss: 1.693249  [12800/60000]
loss: 1.608616  [19200/60000]
loss: 1.583267  [25600/60000]
loss: 1.634061  [32000/60000]
loss: 1.590620  [38400/60000]
loss: 1.600522  [44800/60000]
loss: 1.731807  [51200/60000]
loss: 1.358284  [57600/60000]
Test Error:
 Accuracy: 51.9%, Avg loss: 0.024053

Epoch 6
-------------------------------
loss: 1.761160  [    0/60000]
loss: 1.724018  [ 6400/60000]
loss: 1.536310  [12800/60000]
loss: 1.454606  [19200/60000]
loss: 1.403009  [25600/60000]
loss: 1.506267  [32000/60000]
loss: 1.446292  [38400/60000]
loss: 1.480377  [44800/60000]
loss: 1.607209  [51200/60000]
loss: 1.229744  [57600/60000]
Test Error:
 Accuracy: 52.8%, Avg loss: 0.022136

Epoch 7
-------------------------------
loss: 1.642650  [    0/60000]
loss: 1.617023  [ 6400/60000]
loss: 1.415306  [12800/60000]
loss: 1.344459  [19200/60000]
loss: 1.186358  [25600/60000]
loss: 1.356849  [32000/60000]
loss: 1.264209  [38400/60000]
loss: 1.301946  [44800/60000]
loss: 1.426484  [51200/60000]
loss: 1.094530  [57600/60000]
Test Error:
 Accuracy: 59.2%, Avg loss: 0.019390

Epoch 8
-------------------------------
loss: 1.444386  [    0/60000]
loss: 1.432928  [ 6400/60000]
loss: 1.284451  [12800/60000]
loss: 1.270481  [19200/60000]
loss: 1.000330  [25600/60000]
loss: 1.257163  [32000/60000]
loss: 1.160903  [38400/60000]
loss: 1.218728  [44800/60000]
loss: 1.348202  [51200/60000]
loss: 1.015290  [57600/60000]
Test Error:
 Accuracy: 60.7%, Avg loss: 0.018154

Epoch 9
-------------------------------
loss: 1.356038  [    0/60000]
loss: 1.363030  [ 6400/60000]
loss: 1.206954  [12800/60000]
loss: 1.211904  [19200/60000]
loss: 0.924623  [25600/60000]
loss: 1.188801  [32000/60000]
loss: 1.095822  [38400/60000]
loss: 1.163610  [44800/60000]
loss: 1.283653  [51200/60000]
loss: 0.963228  [57600/60000]
Test Error:
 Accuracy: 62.2%, Avg loss: 0.017268

Epoch 10
-------------------------------
loss: 1.282992  [    0/60000]
loss: 1.308621  [ 6400/60000]
loss: 1.140574  [12800/60000]
loss: 1.167652  [19200/60000]
loss: 0.875548  [25600/60000]
loss: 1.133268  [32000/60000]
loss: 1.047127  [38400/60000]
loss: 1.120583  [44800/60000]
loss: 1.230095  [51200/60000]
loss: 0.924153  [57600/60000]
Test Error:
 Accuracy: 63.5%, Avg loss: 0.016563

Done!
"""