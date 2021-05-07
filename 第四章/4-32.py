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

#2重ループで捜査して動詞のbaseだけ格納
verb_bases = [
    morph["base"] 
    for sentence in sentences 
    for morph in sentence 
    if morph["pos"] == "動詞"
]

#10行目まで表示
for i in range(10):
    print(verb_bases[i])

"""
生れる
つく
する
泣く
する
いる
始める
見る
聞く
捕える
"""