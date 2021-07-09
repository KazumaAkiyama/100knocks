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
    path = "./split",
    train = "train",
    validation = "dev",
    test = "test",
    format = "tsv",
    fields = [("en", EN), ("ja", JA)]
)

JA.build_vocab(train, min_freq = 2)
EN.build_vocab(train, min_freq = 2)

train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (train, valid, test),
    batch_sizes = (16, 16, 16),
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
    input_size_encoder = len(pre_JA.vocab)
    encoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    encoder_dropout = float(0.5)

    encoder = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)

    input_size_decoder = len(pre_EN.vocab)
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    decoder_dropout = float(0.5)
    output_size = len(pre_EN.vocab)

    decoder = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, decoder_dropout, output_size).to(device)

    model = Seq2Seq(encoder, decoder).to(device)

    model.load_state_dict(torch.load("model_weight.pth"))

    PAD_IDX = EN.vocab.stoi['<pad>']
 
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    optimizer.load_state_dict(torch.load("optimizer_model.pth"))

    num_epochs = 10
    clip = 1

    model.train(True)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_loss = evaluate(model, test_iter, criterion)

if __name__ == "__main__":
    main()
