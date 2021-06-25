from RNN_9_81 import newDataset, Sampler, DescendingSampler, gen_loader, gen_descending_loader
from Train_9_82 import Task, Trainer, Predictor
from RNN_9_81 import newDataset, Sampler, DescendingSampler, RNN, gen_loader, gen_descending_loader
from Train_9_82 import Task, Trainer, Predictor
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import pickle
from gensim.models import KeyedVectors

class BiRNN(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super(BiRNN, self).__init__()
        self.embed = nn.Embedding(v_size, e_size)
        #層を増やして，双方向化のフラグをTrueにする
        self.rnn = nn.LSTM(e_size, h_size, num_layers = 2, bidirectional = True)
        self.out = nn.Linear(h_size * 2, c_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                nn.init.uniform_(param, -0.1, 0.1)
        nn.init.uniform_(self.out.weight, -0.1, 0.1)

    def forward(self, batch, h=None):
        x = self.embed(batch['inputs'])
        x = pack(x, batch['lengths'])
        x, (h, c) = self.rnn(x, h)
        h = self.out(h)
        return h

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

trained_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

def init_embed(embed):
    for i, token in enumerate(word_list):
        if token in trained_vectors:
            embed.weight.data[i] = torch.from_numpy(trained_vectors[token])
    return embed

train_dataset = newDataset(train_ids, train_labels)
valid_dataset = newDataset(valid_ids, valid_labels)
test_dataset = newDataset(test_ids, test_labels)

model = BiRNN(len(word_dict), 300, 128, 4)

init_embed(model.embed)

loaders = (
    gen_descending_loader(train_dataset, 128),
    gen_descending_loader(valid_dataset, 128),
)
task = Task()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
trainer = Trainer(model, loaders, task, optimizer, 10)
trainer.train()

#正解率の計算をする
def accuracy(true, pred):
    return np.mean([t == p for t, p in zip(true, pred)])

predictor = Predictor(model, gen_loader(train_dataset, 1))
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_labels, pred))

predictor = Predictor(model, gen_loader(test_dataset, 1))
pred = predictor.predict()
print('評価データでの正解率 :', accuracy(test_labels, pred))

"""
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
epoch 1, train_loss:1.1292495205768271, valid_loss:1.0101736069283682
epoch 2, train_loss:0.8174852227875312, valid_loss:0.7065479045001308
epoch 3, train_loss:0.5824349788022366, valid_loss:0.5465920687963565
学習データでの正解率 : 0.8380606514414077
評価データでの正解率 : 0.8049101796407185
"""