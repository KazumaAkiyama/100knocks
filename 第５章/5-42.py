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
            print(''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morph_list]),
                  ''.join([morph.surface if morph.pos != '記号' else '' for morph in sentence[int(chunk.dst)].morph_list]), sep='\t')

"""
人工知能        語
じんこうちのう  語
AI      エーアイとは
エーアイとは    語
計算    という
という  道具を
概念と  道具を
コンピュータ    という
という  道具を
道具を  用いて
用いて  研究する
知能を  研究する
研究する        計算機科学
計算機科学      の
の      一分野を
一分野を        指す
指す    語
語      研究分野とも
"""