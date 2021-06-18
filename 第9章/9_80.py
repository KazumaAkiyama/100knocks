import re
import spacy
import torch
from collections import Counter
import pickle

nlp = spacy.load('en_core_web_sm')
categories = ['b', 't', 'e', 'm']
category_names = ['business', 'science and technology', 'entertainment', 'health']

#テキストを単語トークンに分ける
def tokenize(text):
    text = re.sub(r'\s+', ' ', text)
    text_tokens = nlp.make_doc(text)
    words = [token.text for token in text_tokens]
    return words

#ファイルからデータセットを読み込む，ラベルはここでテンソル形式にしておく
def read_dataset(filename):
    with open(filename) as f:
        dataset = f.read().splitlines()
    dataset = [line.split('\t') for line in dataset]
    dataset_labels = [categories.index(line[0]) for line in dataset]
    dataset_headlines = [tokenize(line[1]) for line in dataset]
    return dataset_headlines, torch.tensor(dataset_labels, dtype=torch.long)

#見出し文とラベルを読み込む
train_headlines, train_labels = read_dataset('train.txt')
valid_headlines, valid_labels = read_dataset('valid.txt')
test_headlines, test_labels = read_dataset('test.txt')

#単語数をカウントするためのカウンターを生成
counter = Counter([word for headline in train_headlines for word in headline])

#カウントが2以上の単語のみ追加
train_words = [word for word, frequency in counter.most_common() if frequency > 1]

#出現回数2回未満のものに0をつけるための要素を追加
word_list = ['[unknown]'] + train_words

#単語と頻度順位の辞書を作成
word_dict = {x:n for n, x in enumerate(word_list)}

#見出し文をIDの列にする，リストに入ってないやつは出現回数2回未満なので先頭要素のインデックスをとることで0が入る
def headline_to_ids(headline):
    return torch.tensor([word_dict[x if x in word_dict else '[unknown]'] for x in headline], dtype=torch.long)

#データセットの各文に対してIDへの変換を行う
def dataset_to_ids(dataset):
    return [headline_to_ids(x) for x in dataset]

#データセットからIDへの変換を実行
train_ids = dataset_to_ids(train_headlines)
valid_ids = dataset_to_ids(valid_headlines)
test_ids = dataset_to_ids(test_headlines)

#ピックルで保存
with open('train.ids.pickle', 'wb') as f:
    pickle.dump(train_ids, f)
with open('train.label.pickle', 'wb') as f:
    pickle.dump(train_labels, f)

with open('valid.ids.pickle', 'wb') as f:
    pickle.dump(valid_ids, f)
with open('valid.label.pickle', 'wb') as f:
    pickle.dump(valid_labels, f)

with open('test.ids.pickle', 'wb') as f:
    pickle.dump(test_ids, f)
with open('test.label.pickle', 'wb') as f:
    pickle.dump(test_labels, f)

with open('word_dict.pickle', 'wb') as f:
    pickle.dump(word_dict, f)

