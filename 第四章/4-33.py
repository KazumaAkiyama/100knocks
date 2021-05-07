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
#AのBという形の名詞句を格納するリスト
AnoB_list = []

#「名詞+の+名詞」という形で連続しているmorphemeを探して，一致するものを連結してリストに格納する
for morphs in sentences:
    for i in range(1, len(morphs)-1):
        if morphs[i-1]["pos"] == "名詞" and morphs[i]["surface"] == "の" and morphs[i+1]["pos"] == "名詞": 
            AnoB_list.append(morphs[i-1]["surface"] + morphs[i]["surface"]+morphs[i+1]["surface"])

#10行目まで表示
for i in range(10):
    print(AnoB_list[i])

"""
彼の掌
掌の上
書生の顔
はずの顔
顔の真中
穴の中
書生の掌
掌の裏
何の事
肝心の母親
"""