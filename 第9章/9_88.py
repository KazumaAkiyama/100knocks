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
import optuna

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

class CNNDataset(newDataset):
    def collate(self, xs):
        max_seq_len = max([x['lengths'] for x in xs])
        inputs = [torch.cat([x['inputs'], torch.zeros(max_seq_len - x['lengths'], dtype=torch.long)], dim=-1) for x in xs]
        inputs = torch.stack(src)
        mask = [[1] * x['lengths'] + [0] * (max_seq_len - x['lengths']) for x in xs]
        mask = torch.tensor(mask, dtype=torch.long)
        return {
            'inputs':inputs,
            'labels':torch.tensor([x['labels'] for x in xs]),
            'mask':mask,
        }

class CNN(nn.Module):
    def __init__(self, v_size, e_size, h_size, c_size, dropout=0.2):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(v_size, e_size)
        #カーネルを3に.paddingを1で指定するとpaddingトークンが両端に挿入される
        self.conv = nn.Conv2d(e_size, h_size, (3, emb_size), 1, (1, 0))
        self.act = nn.ReLU()
        self.out = nn.Linear(h_size, c_size)
        self.dropout = nn.Dropout(dropout)
        #重みを正規分布で，バイアスを０で初期化
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

train_dataset = CNNDataset(train_ids, train_labels)
valid_dataset = CNNDataset(valid_ids, valid_labels)
test_dataset = CNNDataset(test_ids, test_labels)

#正解率の計算をする
def accuracy(true, pred):
    return np.mean([t == p for t, p in zip(true, pred)])

def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e_size = int(trial.suggest_discrete_uniform('e_size', 100, 4000, 100))
    h_size = int(trial.suggest_discrete_uniform('h_size', 100, 1000, 100))
    dropout = float(trial.suggest_discrete_uniform('dropout', 0.1, 0.5, 0.1))
    model = CNN(len(word_dict), e_size, h_size, 4, dropout)

    loaders = (
        gen_descending_loader(train_dataset, 128),
        gen_descending_loader(valid_dataset, 128),
    )
    task = Task()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainer = Trainer(model, loaders, task, optimizer, 10)
    trainer.train()
    trainer.test()
    predictor = Predictor(model, gen_loader(train_dataset, 1))
    pred = predictor.predict()
    return 1- accuracy(test_labels, pred)

study = optuna.create_study()
study.optimize(objective, n_trials = 100)
print(study.best_params)

model = CNN(len(word_dict), study.best_params["e_size"], study.best_params["h_size"], 4, study.best_params["dropout"])

loaders = (
    gen_descending_loader(train_dataset, 128),
    gen_descending_loader(valid_dataset, 128),
)
task = Task()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
trainer = Trainer(model, loaders, task, optimizer, 10)
trainer.train()



predictor = Predictor(model, gen_loader(train_dataset, 1))
pred = predictor.predict()
print('学習データでの正解率 :', accuracy(train_labels, pred))

predictor = Predictor(model, gen_loader(test_dataset, 1))
pred = predictor.predict()
print('評価データでの正解率 :', accuracy(test_labels, pred))

"""
epoch 1, train_loss:1.1201249768508181, valid_loss:0.9579701952831575
epoch 2, train_loss:0.7999270215925541, valid_loss:0.6637113849935906
epoch 3, train_loss:0.5756728831754212, valid_loss:0.5500919902711158
学習データでの正解率 : 0.8440359415949082
評価データでの正解率 : 0.8264071856287425
"""