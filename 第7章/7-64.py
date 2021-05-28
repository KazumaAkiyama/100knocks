from gensim.models import KeyedVectors

#これで単語ベクトルのファイルを読み込めるらしい．ファイルは.binなのでbinary=Ture
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary = True)

#読み取り用ファイル
with open("questions-words.txt") as f:
#書き込み用ファイル
    with open("question-words-similarity.txt", "w") as g:
        for line in f:
            word_list = line.split()
            #ラベルの行はそのまま出力
            if word_list[0] == ":":
                g.write(" ".join(word_list))
                g.write("\n")
            #それ以外の行は類似度が高い単語とその類似度を追加して出力
            else:
                word, similarity = model.most_similar(positive=[word_list[1], word_list[2]], negative=[word_list[0]], topn=1)[0]
                g.write(" ".join(word_list))
                g.write(" " + word + " " + str(similarity) + "\n")

"""
: capital-common-countries
Athens Greece Baghdad Iraq Iraqi 0.6351870894432068
Athens Greece Bangkok Thailand Thailand 0.7137669324874878
Athens Greece Beijing China China 0.7235777974128723
Athens Greece Berlin Germany Germany 0.6734622120857239
Athens Greece Bern Switzerland Switzerland 0.4919748306274414
Athens Greece Cairo Egypt Egypt 0.7527809739112854
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""