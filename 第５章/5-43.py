class Morph:
    def __init__(self, dc):
        self.surface = dc['surface']
        self.base = dc['base']
        self.pos = dc['pos']
        self.pos1 = dc['pos1']

class Chunk:
    def __init__(self, morph_list, dst):
        self.morph_list = morph_list
        self.dst = dst
        self.srcs = []

def read_cabocha(sentence):
    def append_chunk(morph_list):
        if len(morph_list) > 0:
            chunk = Chunk(morph_list, dst)
            chunk_list.append(chunk)
            morph_list = []
        return morph_list

    morph_list = []
    chunk_list = []
    dst = None
    for line in sentence.split('\n'):
        if line == '':
            morph_list = append_chunk(morph_list)
        elif line[0] == '*':
            morph_list = append_chunk(morph_list)
            dst = line.split(' ')[2].rstrip('D')
        else:
            (surface, attr) = line.split('\t')
            attr = attr.split(',')
            lineDict = {
                'surface': surface,
                'base': attr[6],
                'pos': attr[0],
                'pos1': attr[1]
            }
            morph_list.append(Morph(lineDict))

    for i, r in enumerate(chunk_list):
        chunk_list[int(r.dst)].srcs.append(i)
    return chunk_list

with open("ai.ja.txt.parsed") as f:
    sentence_list = f.read().split('EOS\n')
sentence_list = list(filter(lambda x: x != '', sentence_list))
sentence_list = [read_cabocha(sentence) for sentence in sentence_list]

for sentence in sentence_list:
    for chunk in sentence:
        if int(chunk.dst) > -1:
            pre_text = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morph_list])
            pre_pos = [morph.pos for morph in chunk.morph_list]
            post_text = ''.join([morph.surface if morph.pos != '記号' else '' for morph in sentence[int(chunk.dst)].morph_list])
            post_pos = [morph.pos for morph in sentence[int(chunk.dst)].morph_list]
            if '名詞' in pre_pos and '動詞' in post_pos:
                print(pre_text, post_text, sep='\t')

"""
道具を  用いて
知能を  研究する
一分野を        指す
知的行動を      代わって
人間に  代わって
コンピューターに        行わせる
研究分野とも    される
解説で  述べている
佐藤理史は      述べている
次のように      述べている
知的能力を      実現する
コンピュータ上で        実現する
技術ソフトウェアコンピュータシステム    ある
応用例は        ある
推論判断を      模倣する
画像データを    解析して
解析して        検出抽出したりする
パターンを      検出抽出したりする
画像認識等が    ある
1956年に        命名された
ダートマス会議で        命名された
"""