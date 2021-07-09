import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random
import time
import torchtext
from torchtext import data
from torchtext import datasets
from tqdm import tqdm
import re
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenizer_ja(text):
    result = text.replace("▁", "")
    return result.split()

def tokenizer_en(text):
    pattern = re.compile(r" |@@ ")
    return pattern.split(text)

JA = data.Field(sequential = True, tokenize = tokenizer_ja, lower = True)
EN = data.Field(sequential = True, tokenize = tokenizer_en, lower = True)

train, valid, test = data.TabularDataset.splits(
    path = "./",
    train = "train_sub.tsv",
    validation = "valid_sub.tsv",
    test = "test_sub.tsv",
    format = "tsv",
    fields = [("ja", JA), ("en", EN)]
)

JA.build_vocab(train, min_freq = 2)
EN.build_vocab(train, min_freq = 2)

train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (train, valid, test),
    batch_sizes = (16, 16, 16),
    sort_within_batch = False,
    sort_key = lambda x: len(x.ja),
    device = device
)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.tag = True
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.LSTM = nn.LSTM(self.embedding_size, hidden_size, num_layers, dropout = p)
    
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden_state, cell_state) = self.LSTM(embedding)
        return hidden_state, cell_state

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, output_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = nn.Dropout(p)
        self.tag = True
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.LSTM = nn.LSTM(self.embedding_size, hidden_size, num_layers, dropout = p)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden_state, cell_state):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden_state, cell_state) = self.LSTM(embedding, (hidden_state, cell_state))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden_state, cell_state

class Seq2Seq(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Seq2Seq, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, source, target, tfr = 0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(EN.vocab)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        hidden_state_encoder, cell_state_encoder = self.Encoder(source)
        x = target[0]
        for i in range(1, target_len):
            output, hidden_state_decoder, cell_state_decoder = self.Decoder(x, hidden_state_encoder, cell_state_encoder)
            outputs[i] = output
            best_guess = output.argmax(1)
            x = target[i] if random.random() < tfr else best_guess
        return outputs

def train(model, iterator, optimizer, criterion, clip, writer):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator)):
        src = batch.ja
        trg = batch.en
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.requires_grad_(True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        writer.add_scalar("Training loss", loss.item(), i)
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.ja
            trg = batch.en
            output = model(src, trg, 0)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

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

    PAD_IDX = EN.vocab.stoi['<pad>']
 
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    clip = 1

    model.train(True)

    writer = SummaryWriter(log_dir="./train_logs")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, criterion, clip, writer)
        valid_loss = evaluate(model, valid_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    writer.close()

    test_loss = evaluate(model, test_iter, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    torch.save(model.state_dict(), "model_weight.pth")
    
if __name__ == "__main__":
    main()