import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random
import time
import torchtext
from torchtext import data
from torchtext import datasets
import MeCab
import Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenizer(text):
    return text.split()

JA = data.Field(sequential = True, tokenize = tokenizer, lower = True)
EN = data.Field(sequential = True, tokenize = tokenizer, lower = True)

train, valid, test = data.TabularDataset.splits(
    path = "./",
    train = "train.tsv",
    validation = "valid.tsv",
    test = "test.tsv",
    format = "tsv",
    fields = [("ja", JA), ("en", EN)]
)

JA.build_vocab(train, min_freq = 2)
EN.build_vocab(train, min_freq = 2)

train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (train, valid, test),
    batch_sizes = (2, 2, 2),
    device = device
)

def translate(model, text):
    model.eval()
    wakati = MeCab.Tagger("-Owakati")
    seq = wakati.parse(text).rstrip()
    seq = seq.split()
    seq = [s for s in seq if s in JA.vocab]
    seq = ' '.join(seq)
    output = model(src, "")
    output = output[1:].view(-1, output.shape[-1])
    return ' '.join(output)


def main():
    input_size_encoder = len(JA.vocab)
    encoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    encoder_dropout = float(0.5)

    encoder = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)

    input_size_decoder = len(EN.vocab)
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    decoder_dropout = float(0.5)
    output_size = len(EN.vocab)

    decoder = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, decoder_dropout, output_size).to(device)

    model = Seq2Seq(encoder, decoder).to(device)

    model.load_state_dict(torch.load("model_weight.pth"))

    text = '自然言語処理の100本ノック'
    print (translate(model, text))



if __name__ == "__main__":
    main()
