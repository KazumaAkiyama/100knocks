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

with open("5-44.txt", "w") as f:
    for sentence in sentence_list:
        for chunk in sentence:
            if len(chunk.srcs) > 0:
                pre_morphs = [sentence[int(s)].morph_list for s in chunk.srcs]
                pre_morphs = [list(filter(lambda x: '助詞' in x.pos, pm)) for pm in pre_morphs]
                pre_surface = [[p.surface for p in pm] for pm in pre_morphs]
                pre_surface = list(filter(lambda x: x != [], pre_surface))
                pre_surface = [p[0] for p in pre_surface]
                post_base = [morph.base for morph in chunk.morph_list]
                post_pos = [morph.pos for morph in chunk.morph_list]
                if len(pre_surface) > 0 and '動詞' in post_pos:
                    print(post_base[0], ' '.join(pre_surface), sep='\t', file=f)

"""
akiyama@LAPTOP-8R9KUU89:~/100knocks/第５章$ cat ./5-44.txt | grep '行う' | sort | uniq -c | sort -nr | head -n 5
      8 行う    を
      1 行う    を に を
      1 行う    まで を に
      1 行う    まで を
      1 行う    は を をめぐって
akiyama@LAPTOP-8R9KUU89:~/100knocks/第５章$ cat ./5-44.txt | grep 'なる' | sort | uniq -c | sort -nr | head -n 5
      3 なる    が と
      2 なる    は に
      2 なる    に
      2 なる    と
      1 異なる  も
akiyama@LAPTOP-8R9KUU89:~/100knocks/第５章$ cat ./5-44.txt | grep '与える' | sort | uniq -c | sort -nr | head -n 5
      1 与える  は に を
      1 与える  が に
      1 与える  が など
"""