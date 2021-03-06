from gensim.models import KeyedVectors

#これで単語ベクトルのファイルを読み込めるらしい．ファイルは.binなのでbinary=Ture
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary = True)

#類似度が高い単語を表示．デフォルトでtopn=10なので特に指定はいらない．
print(model.most_similar("United_States"))

"""
akiyama@LAPTOP-8R9KUU89:~/100knocks/第7章$ python3 7-62.py
[('Unites_States', 0.7877248525619507), 
('Untied_States', 0.7541370391845703), 
('United_Sates', 0.74007248878479), 
('U.S.', 0.7310774326324463), 
('theUnited_States', 0.6404393911361694), 
('America', 0.6178410053253174), 
('UnitedStates', 0.6167312264442444), 
('Europe', 0.6132988929748535), 
('countries', 0.6044804453849792), 
('Canada', 0.6019070148468018)]
"""