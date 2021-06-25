from RNN_9_81 import newDataset, Sampler, DescendingSampler, RNN, gen_loader, gen_descending_loader
from Train_9_82 import Task, Trainer, Predictor
from RNN_9_81 import newDataset, Sampler, DescendingSampler, RNN, gen_loader, gen_descending_loader
from Train_9_82 import Task, Trainer, Predictor
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.optim as optim
import pickle
from gensim.models import KeyedVectors

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

model = RNN(len(word_dict), 300, 128, 4)

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
epoch 1, train_loss:0.7509230666131859, valid_loss:0.5375294953584671
epoch 2, train_loss:0.7565749380961958, valid_loss:1.0035248339176177
epoch 3, train_loss:0.3542013850556799, valid_loss:0.44220863878726957
epoch 4, train_loss:0.44197026361902075, valid_loss:0.4919645845890045
epoch 5, train_loss:0.24630768731775055, valid_loss:0.3593780279159546
epoch 6, train_loss:0.1615781494113336, valid_loss:0.3575172483921051
epoch 7, train_loss:0.11395291719271476, valid_loss:0.36075035482645035
epoch 8, train_loss:0.08282777251877699, valid_loss:0.3837740033864975
epoch 9, train_loss:0.05891009454940816, valid_loss:0.39766719937324524
epoch 10, train_loss:0.04517540664423302, valid_loss:0.417317533493042
学習データでの正解率 : 0.8554249344814676
評価データでの正解率 : 0.8234131736526946
"""