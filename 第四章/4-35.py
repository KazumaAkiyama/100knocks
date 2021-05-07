import collections

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
#2重ループをまわしてmorphのsurfaceを全部格納し，surfaceのリストをつくる
surfaces = [
    morph["surface"]
    for sentence in sentences
    for morph in sentence
]
#Counterで出現頻度の辞書を作成後，most_commonで出現頻度順に並べたリストにする
freq_of_appear = collections.Counter(surfaces).most_common()

#10行目まで表示
for _,(word, freq) in zip(range(10), freq_of_appear):
    print(surface, freq)

"""
の 9194
。 7486
て 6868
、 6772
は 6420
に 6243
を 6071
と 5508
が 5337
た 3988
"""