from transformers import *

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

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def read_for_bert(filename):
    with open(filename) as f:
        dataset = f.read().splitlines()
    dataset = [line.split('\t') for line in dataset]
    dataset_labels = [categories.index(line[0]) for line in dataset]
    dataset_headlines = [torch.tensor(tokenizer.encode(line[1]), dtype=torch.long) for line in dataset]
    return dataset_headlines, torch.tensor(dataset_labels, dtype=torch.long)

train_headlines, train_labels = read_dataset('train.txt')
valid_headlines, valid_labels = read_dataset('valid.txt')
test_headlines, test_labels = read_dataset('test.txt')

class BertDataset(Dataset):
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

train_dataset = BertDataset(train_ids, train_labels)
valid_dataset = BertDataset(valid_ids, valid_labels)
test_dataset = BertDataset(test_ids, test_labels)

class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-cased', num_labels=4)
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-cased', config=config)

    def forward(self, batch):
        x = self.bert(batch['inputs'], attention_mask=batch['mask'])
        return x[0]

loaders = (
    gen_descending_loader(train_dataset, 128),
    gen_descending_loader(valid_dataset, 128),
)
task = Task()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
trainer = Trainer(model, loaders, task, optimizer, 5)
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