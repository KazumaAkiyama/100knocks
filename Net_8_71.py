import pickle
import torch
import torch.nn as nn

with open('train.feature.pickle', 'rb') as f:
    train_vectors = pickle.load(f)
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(300, 4)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x
    
model = Net()
torch.save(model, 'model.pth')

x = model(train_vectors[0])
x = torch.softmax(x, dim=-1)
print(x)

x = model(train_vectors[:4])
x = torch.softmax(x, dim=-1)
print(x)

"""
tensor([0.2376, 0.2169, 0.2739, 0.2715], grad_fn=<SoftmaxBackward>)
tensor([[0.2376, 0.2169, 0.2739, 0.2715],
        [0.2262, 0.2163, 0.2453, 0.3122],
        [0.2137, 0.2471, 0.2716, 0.2676],
        [0.2495, 0.2456, 0.2362, 0.2687]], grad_fn=<SoftmaxBackward>)
"""
