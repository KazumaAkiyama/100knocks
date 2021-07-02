from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

#デバイスの切り替え
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#文の開始・終了用のトークンは別に用意しておく
SOS_token = 0
EOS_token = 1

#言語ごとに"単語→ID"，"単語→登場回数", "ID→単語"の辞書を作成する
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        #SOSとEOSの分があるからスタートは2
        self.n_words = 2

    #引数として文を渡すと単語に分解して辞書に登録してくれる
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    #単語を辞書に登録
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

#文字列の正規化
def normalizeString(s):
    t = s
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    if len(s.replace(' ','')): # ascii文字以外
      return s
    #ascii文字はそのまま返す
    return t

#2言語のファイル名を受け取って，行ごとに分割してペアリストをつくる
def readLangs(lang1, lang2):
    print("Reading lines...")
    with open(lang1) as f:
      lines1 = f.readlines()
    with open(lang2) as f:
      lines2 = f.readlines()
    pairs = []
    for l1,l2 in zip(lines1,lines2):
      l1 = normalizeString(l1.rstrip('\n'))
      l2 = normalizeString(l2.rstrip('\n'))
      pairs.append([l1,l2])
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

#単語数がMAX_LENGTH以下のペアリストをつくる
MAX_LENGTH = 10
def filterPair(p): 
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

#学習用のトークナイズされたファイルのディレクトリ
train_ja = 'kftt-data-1.0/data/tok/kyoto-train.cln.ja'
train_en = 'kftt-data-1.0/data/tok/kyoto-train.cln.en'

#2言語のファイルディレクトリを受け取ってデータとして処理する
def prepareData(lang1, lang2):
    #テキストファイルを読み取って行分割とペア分割をする
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    
    #テキストに文字列長でフィルタをかける
    pairs = filterPairs(pairs)

    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    
    #ペアの文を追加していって各言語のリストをつくる
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData(train_ja, train_en)
print(random.choice(pairs))

#エンコーダー
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    #隠れ層の初期化
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#アテンションデコーダー
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        #アテンションの重みを正規化
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #アテンションの重みとエンコーダーの行列積をとる
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        #アテンションをかけたエンコーダの出力
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    #隠れそうの重みを初期化
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#文を渡したらインデックスのリストにしてくれる
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

#文を渡したらEOSを文末に追加してテンソルにしてくれる
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

#ペアデータを渡したらそれぞれEOSを追加してテンソルにしてくれる
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

#デコーダーの出力は半分だけ使用する
teacher_forcing_ratio = 0.5

#学習に必要な各要素を渡して学習させる
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #エンコーダーの出力を初期化しておく
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    #文を入力していってエンコーダの学習をまわす
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    #正解データを次の入力にまわすかどうか決める
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math

#時間を分秒の単位に直す
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#経過時間と残りの推定時間を計算する
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

#イテレータを回して学習を行う
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    #タイマーを開始
    start = time.time()
    print_loss_total = 0

    #エンコーダとデコーダーのオプティマイザを宣言
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #ランダムなペアをとってきてテンソルにする
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    #損失関数の定義
    criterion = nn.NLLLoss()

    #ペアデータから入力・正解データを取り出し，trainにわたす
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        #print_everyごと(ここでは5000)に経過時間やロスの出力を行う
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

#評価用関数．各ステップでデコーダーが出した予測を自身にフィードバックする
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        #デコーダーが出力(予測)した単語を文字リストに追加していく．EOSなら終了
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

#ランダムにとってきたペアデータについて，入力・正解・予測データを出力
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder, attn_decoder, 75000, print_every=5000)

evaluateRandomly(encoder,attn_decoder,10)

torch.save(encoder.state_dict(), 'encoder_weights.pth')
torch.save(attn_decoder.state_dict(), 'attn_decoder_weight.pth')

"""
Reading lines...
Read 329882 sentence pairs
Trimmed to 67831 sentence pairs
Counting words...
Counted words:
kftt-data-1.0/data/tok/kyoto-train.cln.ja 34123
kftt-data-1.0/data/tok/kyoto-train.cln.en 32109
['三 町 ( 高山 市 )', 'sanmachi takayama city ']
1m 54s (- 26m 38s) (5000 6%) 5.4597
3m 35s (- 23m 22s) (10000 13%) 5.1825
5m 33s (- 22m 13s) (15000 20%) 4.9647
7m 32s (- 20m 44s) (20000 26%) 4.7513
9m 30s (- 19m 1s) (25000 33%) 4.5814
11m 22s (- 17m 4s) (30000 40%) 4.5173
13m 20s (- 15m 15s) (35000 46%) 4.3782
15m 19s (- 13m 24s) (40000 53%) 4.3756
17m 9s (- 11m 26s) (45000 60%) 4.2520
18m 59s (- 9m 29s) (50000 66%) 4.2188
20m 59s (- 7m 38s) (55000 73%) 4.1170
22m 47s (- 5m 41s) (60000 80%) 4.0610
24m 20s (- 3m 44s) (65000 86%) 4.0637
26m 1s (- 1m 51s) (70000 93%) 3.9671
28m 2s (- 0m 0s) (75000 100%) 3.8737
> 建久 年中 検田 帳
= the kenkyu era cadastral survey records
< the and in in <EOS>

> 土岐 光 衡 の 長男 。
= the eldest son of mitsuhira toki
< the eldest son of the <EOS>

> キリン （ 「 清水 」 「 未来 」 ）
= giraffe kiyomizu and mirai
<  the  <EOS>

> 乾物 の 一種 。
= it is a type of dry food .
< a are of of <EOS>

> 薬味 と し て 用い て い る 。
= it is used as a seasoning .
< it is a . . <EOS>

> 大神 神社
= omiwa jinja shrine
< shrine jinja shrine <EOS>

> （ この 項 未 執筆 ）
=  this section is not written yet
<   the <EOS>

> チベット 語 訳
= translation into tibetan
< translation translation <EOS>

> （ → “ 主な 会派 ” ）
=  major factions
<  the  <EOS>

> a
= a kiritsubo tsubosenzai
< a a a a a a a a <EOS>
"""