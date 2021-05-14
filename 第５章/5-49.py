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

def convert(sentence):
    pl, nl = [], [chunk for chunk in sentence if '名詞' in [morph.pos for morph in chunk.morph_list]]
    for i in range(len(nl) - 1):
        st1 = [''.join([morph.surface if morph.pos != '名詞' else 'X' for morph in nl[i].morph_list])]
        for e in nl[i + 1:]:
            dst, p = nl[i].dst, []
            st2 = [''.join([morph.surface if morph.pos != '名詞' else 'Y' for morph in e.morph_list])]
            while int(dst) != -1 and dst != sentence.index(e):
                p.append(sentence[int(dst)])
                dst = sentence[int(dst)].dst
            if len(p) < 1 or p[-1].dst != -1:
                mid = [''.join([morph.surface for morph in chunk.morph_list if morph.pos != '記号']) for chunk in p]
                pl.append(st1 + mid + ['Y'])
            else:
                mid, dst = [], e.dst
                while not sentence[int(dst)] in p:
                    mid.append(''.join([m.surface for morph in sentence[int(dst)].morph_list if morph.pos != '記号']))
                    dst = sentence[int(dst)].dst
                ed = [''.join([morph.surface for morph in sentence[int(dst)].morph_list if morph.pos != '記号'])]
                pl.append([st1, st2 + mid, ed])
    return pl

with open("ai.ja.txt.parsed") as f:
    sentence_list = f.read().split('EOS\n')
sentence_list = list(filter(lambda x: x != '', sentence_list))
sentence_list = [read_cabocha(sentence) for sentence in sentence_list]

for sentence in sentence_list:
    pl = (convert(sentence))
    for p in pl:
        if isinstance(p[0], str):
            print(' -> '.join(p))
        else:
            print(p[0][0], ' -> '.join(p[1]), p[2][0], sep=' | ')

"""
Xか -> つくのでしょうかと -> 発言し -> 答えている -> Y
Xか -> つくのでしょうかと -> 発言し -> 答えている -> Y
Xか -> つくのでしょうかと -> 発言し -> 答えている -> Y
Xか -> つくのでしょうかと -> 発言し -> 答えている -> Y
Xか -> つくのでしょうかと -> 発言し -> 答えている -> Y
Xが -> つくのでしょうかと -> 発言し -> 答えている -> Y
Xが -> つくのでしょうかと -> 発言し -> 答えている -> Y
Xが -> つくのでしょうかと -> 発言し -> 答えている -> Y
Xが -> つくのでしょうかと -> 発言し -> 答えている -> Y
つくXでしょうか？」と -> 発言し -> 答えている -> Y
つくXでしょうか？」と -> 発言し -> 答えている -> Y
つくXでしょうか？」と -> 発言し -> 答えている -> Y
Xし、 -> 答えている -> Y
Xし、 -> 答えている -> Y
Xは -> 答えている -> Y
"""