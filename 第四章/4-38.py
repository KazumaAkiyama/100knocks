import collections
import matplotlib.pyplot as plt
import japanize_matplotlib

#形態素解析結果読み込み用の関数
def read_mecab():
    sentences = []
    morphs = []
    with open("neko.txt.mecab", 'r') as f:
        for line in f:
            if line == "\n":
                continue
            if  line == "EOS\n":
                sentences.append(morphs)
                morphs = []
            else:
                (surface, attr) = line.split("\t")
                if surface != "":
                    attr = attr.split(",")
                    morph = {"surface":surface, "base":attr[6], "pos":attr[0], "pos1":attr[1]}
                    morphs.append(morph)
    return sentences

sentences = read_mecab()
#2重ループをまわしてmorphのbaseを全部格納し，baseのリストをつくる
words = [
    morph["base"]
    for sentence in sentences
    for morph in sentence 
]
#Counterで出現頻度の辞書を作成後，most_commonで出現頻度順に並べたリストにする
word_freq = collections.Counter(words).most_common()
#出現頻度のリストを作成
freqs = [freq for _,freq in word_freq]

#freqsを用いてヒスグラムを作成
plt.xlabel("出現頻度")
plt.ylabel("単語の種類数")
plt.hist(freqs, bins=100, range=(1,100))
plt.show()