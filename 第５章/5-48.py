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
        text = []
        if '名詞' in [s.pos for s in chunk.morph_list] and int(chunk.dst) != -1:
            current_chunk = chunk
            text.append(''.join([chunk.surface for chunk in current_chunk.morph_list]))
            next_chunk = sentence[int(current_chunk.dst)]
            while int(current_chunk.dst) != -1:
                text.append(''.join([chunk.surface for chunk in next_chunk.morph_list]))
                current_chunk = next_chunk
                next_chunk = sentence[int(next_chunk.dst)]
            print(*text, sep=' -> ')

"""
人工知能 -> 語。 -> 研究分野」とも -> される。
（じんこうちのう、、 -> 語。 -> 研究分野」とも -> される。
AI -> 〈エーアイ〉）とは、 -> 語。 -> 研究分野」とも -> される。
〈エーアイ〉）とは、 -> 語。 -> 研究分野」とも -> される。
「『計算 -> （）』という -> 道具を -> 用いて -> 研究する -> 計算機科学 -> （）の -> 一分野」を -> 指す -> 語。 -> 研究分野 」とも -> される。
概念と -> 道具を -> 用いて -> 研究する -> 計算機科学 -> （）の -> 一分野」を -> 指す -> 語。 -> 研究分野」とも -> される。
『コンピュータ -> （）』という -> 道具を -> 用いて -> 研究する -> 計算機科学 -> （）の -> 一分野」を -> 指す -> 語。 -> 研 究分野」とも -> される。
道具を -> 用いて -> 研究する -> 計算機科学 -> （）の -> 一分野」を -> 指す -> 語。 -> 研究分野」とも -> される。
『知能』を -> 研究する -> 計算機科学 -> （）の -> 一分野」を -> 指す -> 語。 -> 研究分野」とも -> される。
"""