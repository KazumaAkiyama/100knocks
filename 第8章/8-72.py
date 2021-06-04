from Net_8_71 import Net
import pickle
import torch
import torch.nn as nn

with open('train.feature.pickle', 'rb') as f:
    train_vectors = pickle.load(f)
    
with open('train.label.pickle', 'rb') as f:
    train_labels = pickle.load(f)
  
model = torch.load('model.pth')

criterion = nn.CrossEntropyLoss()

output = model(train_vectors[:1])
target = train_labels[:1]
loss = criterion(output, target)
model.zero_grad()
loss.backward()
print('損失：', loss.item())
print('勾配：')
print(model.fc.weight.grad)

output = model(train_vectors[:4])
target = train_labels[:4]
loss = criterion(output, target)
model.zero_grad()
loss.backward()
print('損失：', loss.item())
print('勾配：')
print(model.fc.weight.grad)

"""
損失： 1.4709620475769043
勾配：
tensor([[ 0.0949,  0.0029,  0.0571,  ..., -0.0654, -0.0380,  0.0859],
        [-0.0334, -0.0010, -0.0201,  ...,  0.0230,  0.0133, -0.0302],
        [-0.0292, -0.0009, -0.0176,  ...,  0.0201,  0.0117, -0.0264],
        [-0.0324, -0.0010, -0.0195,  ...,  0.0223,  0.0129, -0.0293]])
損失： 1.3156459331512451
勾配：
tensor([[ 0.0167, -0.0521,  0.0724,  ...,  0.0024, -0.0602,  0.0523],
        [-0.0066,  0.0186, -0.0255,  ..., -0.0012,  0.0215, -0.0190],
        [-0.0047,  0.0160, -0.0235,  ..., -0.0007,  0.0190, -0.0166],
        [-0.0054,  0.0176, -0.0234,  ..., -0.0005,  0.0197, -0.0167]])
"""