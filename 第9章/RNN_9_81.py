import random as rd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import pickle

with open('train.ids.pickle', 'rb') as f:
    train_ids = pickle.load(f)
    
with open('train.label.pickle', 'rb') as f:
    train_labels = pickle.load(f)

with open('test.ids.pickle', 'rb') as f:
    test_ids = pickle.load(f)
    
with open('test.label.pickle', 'rb') as f:
    test_labels = pickle.load(f)

with open('word_dict.pickle', 'rb') as f:
    word_dict = pickle.load(f)

#データセット
class newDataset(Dataset):
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

#パディングは系列長を揃えるために0詰めを行うもの
    def collate(self, xs):
        return {
            'inputs':pad([x['inputs'] for x in xs]),
            'labels':torch.stack([x['labels'] for x in xs], dim=-1),
            'lengths':torch.stack([x['lengths'] for x in xs], dim=-1)
        }

train_dataset = newDataset(train_ids, train_labels)
test_dataset = newDataset(test_ids, test_labels)

#データセットのバッチ分割を行う
class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, width, shuffle = False):
        self.dataset = dataset
        self.width = width
        self.shuffle = shuffle
        if not shuffle:
            self.indices = torch.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(len(self.dataset))
        index = 0
        while index < len(self.dataset):
            yield self.indices[index : index + self.width]
            index += self.width

#インデックスの長さの降順に並べ替える
class DescendingSampler(Sampler):
    def __init__(self, dataset, width, shuffle = False):
        assert not shuffle
        super().__init__(dataset, width, shuffle)
        self.indices = self.indices[self.dataset.lengths[self.indices].argsort(descending=True)]

#バッチ分割したデータを渡してデータローダーを生成する
def gen_loader(dataset, width, sampler=Sampler, shuffle=False, num_workers=8):
    return DataLoader(
        dataset, 
        batch_sampler = sampler(dataset, width, shuffle),
        collate_fn = dataset.collate,
        num_workers = num_workers,
    )
#インデックスの長さの降順に並べたバッチ分割データを渡してデータローダを生成する
def gen_descending_loader(dataset, width, num_workers=0):
    return gen_loader(dataset, width, sampler = DescendingSampler, shuffle = False, num_workers = num_workers)

#RNNモデル
class RNN(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(v_size, e_size)
        self.rnn = nn.LSTM(e_size, h_size, num_layers = 1)
        self.out = nn.Linear(h_size, c_size)
        self.dropout = nn.Dropout(dropout)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)
        self.out.weight.data.uniform_(-0.1, 0.1)

    def forward(self, batch, h=None):
        x = self.embed(batch['inputs'])
        x = pack(x, batch['lengths'])
        x, (h, c) = self.rnn(x, h)
        h = self.out(h)
        return h.squeeze(0)

model = RNN(len(word_dict), 300, 50, 4)
loader = gen_loader(test_dataset, 10, DescendingSampler, False)
print(model(iter(loader).next()).argmax(dim=-1))