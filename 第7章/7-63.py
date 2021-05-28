from gensim.models import KeyedVectors

#これで単語ベクトルのファイルを読み込めるらしい．ファイルは.binなのでbinary=Ture
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary = True)

#positiveに指定した単語ベクトルを加算，negativeに指定した単語ベクトルを減算して一番近いやつをとってくる
print(model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid']))

"""
akiyama@LAPTOP-8R9KUU89:~/100knocks/第7章$ python3 7-63.py
[('Greece', 0.6898481249809265), 
('Aristeidis_Grigoriadis', 0.5606848001480103), 
('Ioannis_Drymonakos', 0.5552908778190613), 
('Greeks', 0.545068621635437), 
('Ioannis_Christou', 0.5400862693786621), 
('Hrysopiyi_Devetzi', 0.5248444676399231), 
('Heraklio', 0.5207759737968445), 
('Athens_Greece', 0.516880989074707), 
('Lithuania', 0.5166866183280945), 
('Iraklion', 0.5146791934967041)]
"""