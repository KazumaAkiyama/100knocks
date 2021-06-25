from RNN_9_81 import newDataset, Sampler, DescendingSampler, gen_loader, gen_descending_loader
import random as rd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn
import torch.nn.functional as F
import pickle

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

with open('word_list.pickle', 'rb') as f:
    word_list = pickle.load(f)

with open('word_dict.pickle', 'rb') as f:
    word_dict = pickle.load(f)

class CNNDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.lengths = torch.tensor([len(x) for x in inputs])
        self.size = len(inputs)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            'inputs':self.inputs[index],
            'labels':self.labels[index],
            'lengths':self.lengths[index],
        }

    def collate(self, xs):
        max_seq_len = max([x['lengths'] for x in xs])
        inputs = [torch.cat([x['inputs'], torch.zeros(max_seq_len - x['lengths'], dtype=torch.long)], dim=-1) for x in xs]
        inputs = torch.stack(inputs)
        mask = [[1] * x['lengths'] + [0] * (max_seq_len - x['lengths']) for x in xs]
        mask = torch.tensor(mask, dtype=torch.long)
        return {
            'inputs':inputs,
            'labels':torch.tensor([x['labels'] for x in xs]),
            'mask':mask,
        }

train_dataset = CNNDataset(train_ids, train_labels)
valid_dataset = CNNDataset(valid_ids, valid_labels)
test_dataset = CNNDataset(test_ids, test_labels)

class CNN(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(v_size, e_size)
        self.conv = nn.Conv1d(e_size, h_size, 3, padding=1)
        self.act = nn.ReLU()
        self.out = nn.Linear(h_size, c_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embed.weight, 0, 0.1)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.kaiming_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, batch):
        x = self.embed(batch['inputs'])
        x = self.dropout(x)
        x = self.conv(x.transpose(-1, -2))
        x = self.act(x)
        x = self.dropout(x)
        x.masked_fill_(batch['mask'].unsqueeze(-2) == 0, -1e4)
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        x = self.out(x)
        return x

model = CNN(len(word_dict), 300, 128, 4)
loader = gen_loader(test_dataset, 10, DescendingSampler, False)
print(model(iter(loader).next()).argmax(dim=-1))

"""
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
"""