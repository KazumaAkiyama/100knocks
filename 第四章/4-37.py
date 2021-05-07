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
cat_co_occure = []
for sentence in sentences:
    if "猫" in [morph['surface'] for morph in sentence]:
        words = [
            morph["surface"]
            for morph in sentence
            if morph["surface"] != "猫" and morph["pos"] in {"名詞", "動詞", "形容詞", "副詞"}
        ]
        cat_co_occure.extend(words)
cat_co_occure = collections.Counter(cat_co_occure).most_common()

words = [word for word,_ in cat_co_occure[:10]]
freqs = [freq for _,freq in cat_co_occure[:10]]
plt.bar(words, freqs)
plt.xlabel('「猫」と共起頻度の高い上位10語')
plt.ylabel('出現頻度')
plt.show()

for i, key, value in zip(range(10), words, freqs):
    print(key, value)

"""
し 83
事 59
吾輩 58
の 55
いる 46
人間 40
ある 39
する 38
もの 36
よう 3
"""