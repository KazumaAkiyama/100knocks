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
    for i, chunk in enumerate(sentence):
        if 'サ変接続' in [s.pos1 for s in chunk.morph_list] and 'を' in [s.surface for s in chunk.morph_list] and i + 1 < len(sentence) and sentence[i + 1].morph_list[0].pos == '動詞':
            text = ''.join([s.surface for s in chunk.morph_list]) + sentence[i + 1].morph_list[0].base
            if len(chunk.srcs) > 0:
                pre_morphs = [sentence[int(s)].morph_list for s in chunk.srcs]
                pre_morphs_filtered = [list(filter(lambda x: '助詞' in x.pos, pm)) for pm in pre_morphs]
                pre_surface = [[p.surface for p in pm] for pm in pre_morphs_filtered]
                pre_surface = list(filter(lambda x: x != [], pre_surface))
                pre_surface = [p[0] for p in pre_surface]
                pre_text = list(filter(lambda x: '助詞' in [p.pos for p in x], pre_morphs))
                pre_text = [''.join([p.surface for p in pt]) for pt in pre_text]
                if len(pre_surface) > 0:
                    print('\t'.join([text, ' '.join(pre_surface), ' '.join(pre_text)]))

"""
流行を超える    の      一過性の
学習を繰り返す  や      開発や
コンテンツ生成を行う    という  CreativeAIという
開発を行う      の      機械式計算機の
投資全額を上回る        の      政府の
研究を進める    の      第五世代コンピュータの
注目を集める    の      世間の
成功を受ける    の      試みの
知的制御を用いる        の      同様の
進歩を担う      の      科学技術の
精度改善を果たす        から    従来手法からの
専用プログラムを使う    にあたる        解析にあたる
関連性を導き出す        の      情報の
認識能力を持つ  の      人間並みの
注目を集める    の      識者の
普及を受ける    と      発明と
機械学習を組み合わせる  と      神経科学と
研究を行う      という  再構成するという
実験をする      や の   研究や 新技術の
弾圧を併せ持つ  と の   資金力と 人権の
監視社会化を恐れる      による  人工知能による
差別を認める    のみ で ビッグデータ分析のみによる、 融資での
展開を変える    の      部隊の
判断を介す      の      人間の
禁止を求める    の      自動操縦型武器の
運用をめぐる    の      AI兵器の
自律無人艇を使う        の      56隻の
試験を行う      の      世界最大規模の
話をする        の      異界の
"""