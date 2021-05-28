from gensim.models import KeyedVectors
from scipy.stats import spearmanr

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

human_vector_similarity_list = []
with open("combined.csv") as f:
    #ファイルオブジェクトはイテレータとして使えるので一行とばせる
    next(f)
    for line in f:
        #改行コードが入って来ちゃうからstripを適用
        word_smilarity_list = line.split(",")
        word_smilarity_list = [i.strip() for i in word_smilarity_list]
        word_smilarity_list.append(model.similarity(word_smilarity_list[0], word_smilarity_list[1]))
        human_vector_similarity_list.append(word_smilarity_list)

human_similarity_list = [i[2] for i in human_vector_similarity_list]
vector_similarity_list = [i[3] for i in human_vector_similarity_list]
correlation, pvalue = spearmanr(human_similarity_list, vector_similarity_list)

print(correlation)

"""
akiyama@LAPTOP-KVF59R5A:~/100knocks/第７章$ python3 7-66.py
0.6849564489532377
"""