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

for b in sentence_list:
    for m in b:
        if len(m.srcs) > 0:
            pre_morphs = [b[int(s)].morph_list for s in m.srcs]
            pre_morphs_filtered = [list(filter(lambda x: '助詞' in x.pos, pm)) for pm in pre_morphs]
            pre_surface = [[p.surface for p in pm] for pm in pre_morphs_filtered]
            pre_surface = list(filter(lambda x: x != [], pre_surface))
            pre_surface = [p[0] for p in pre_surface]
            post_base = [mo.base for mo in m.morph_list]
            post_pos = [mo.pos for mo in m.morph_list]
            if len(pre_surface) > 0 and '動詞' in post_pos:
                pre_text = list(filter(lambda x: '助詞' in [p.pos for p in x], pre_morphs))
                pre_text = [''.join([p.surface for p in pt]) for pt in pre_text]
                print(post_base[0], ' '.join(pre_surface), ' '.join(pre_text), sep='\t')

"""
用いる  を      道具を
研究    て を   用いて 『知能』を
指す    を      一分野」を
代わる  を に   知的行動を 人間に
行う    て に   代わって コンピューターに
する    と      研究分野」とも
述べる  で は の て     解説で、 佐藤理史は 次のように 述べている。
実現    を で   知的能力を コンピュータ上で
模倣    を      推論・判断を
解析    を      画像データを
検出    て を   解析して パターンを
ある    は が   応用例は 画像認識等が
命名    に で により    1956年に ダートマス会議で ジョン・マッカーシーにより
用いる  を      記号処理を
する    を と   記述を 主体と
使う    で でも 現在では、 意味あいでも
呼ぶ    も      思考ルーチンも
ある    て も   使われている。 ことも
"""