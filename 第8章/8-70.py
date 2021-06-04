import re
import spacy
import torch
from gensim.models import KeyedVectors
import pickle

nlp = spacy.load('en_core_web_sm')
categories = ['b', 't', 'e', 'm']
category_names = ['business', 'science and technology', 'entertainment', 'health']

def tokenize(text):
    text = re.sub(r'\s+', ' ', text)
    text_tokens = nlp.make_doc(text)
    words = [token.text for token in text_tokens]
    return words

def read_dataset(filename):
    with open(filename) as f:
        dataset = f.read().splitlines()
    dataset = [line.split('\t') for line in dataset]
    dataset_labels = [categories.index(line[0]) for line in dataset]
    dataset_headlines = [tokenize(line[1]) for line in dataset]
    return dataset_labels, dataset_headlines

train_labels, train_headlines = read_dataset('train.txt')
valid_labels, valid_headlines = read_dataset('valid.txt')
test_labels, test_headlines = read_dataset('test.txt')

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

def headline_to_vector(headline):
    word_vectors = [torch.tensor(model[word]) for word in headline if word in model]
    return sum(word_vectors) / len(word_vectors)

def dataset_to_vector(headlines):
    return torch.stack([headline_to_vector(headline) for headline in headlines])

train_vectors = dataset_to_vector(train_headlines)
valid_vectors = dataset_to_vector(valid_headlines)
test_vectors = dataset_to_vector(test_headlines)

train_labels = torch.tensor(train_labels)
valid_labels = torch.tensor(valid_labels)
test_labels = torch.tensor(test_labels)

with open('train.feature.pickle', 'wb') as f:
    pickle.dump(train_vectors, f)
with open('train.label.pickle', 'wb') as f:
    pickle.dump(train_labels, f)

with open('valid.feature.pickle', 'wb') as f:
    pickle.dump(valid_vectors, f)
with open('valid.label.pickle', 'wb') as f:
    pickle.dump(valid_labels, f)

with open('test.feature.pickle', 'wb') as f:
    pickle.dump(test_vectors, f)
with open('test.label.pickle', 'wb') as f:
    pickle.dump(test_labels, f)