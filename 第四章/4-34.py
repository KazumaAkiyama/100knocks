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
link_nouns = []

link_noun = []
for sentence in sentences:
    for morph in sentence:
        if morph["pos"] == "名詞":
            link_noun.append(morph["surface"])
        elif len(link_noun) > 1:
            link_nouns.append("".join(link_noun))
            link_noun = []
        else:
            link_noun = []

for i in range(10):
    print(link_nouns[i])

"""
人間中
一番獰悪
時妙
一毛
その後猫
一度
ぷうぷうと煙
邸内
三毛
書生以外
"""