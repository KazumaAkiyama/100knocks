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
words = [
    morph["base"]
    for sentence in sentences
    for morph in sentence 
]
word_freq = collections.Counter(words).most_common()
freqs = [freq for _,freq in word_freq]

plt.xscale("log")
plt.yscale("log")
plt.xlabel("出現頻度順位")
plt.ylabel("出現頻度")
plt.plot(range(len(freqs)), freqs)
plt.show()