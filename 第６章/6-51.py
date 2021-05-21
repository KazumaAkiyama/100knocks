import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#データをリストとして読み込んでおく
with open("train.txt") as f:
    train_data = f.readlines()
    train_list = [line.split("\t") for line in train_data]
with open("valid.txt") as f:
    valid_data = f.readlines()
    valid_list = [line.split("\t") for line in valid_data]
with open("test.txt") as f:
    test_data = f.readlines()
    test_list = [line.split("\t") for line in test_data]

#CountVectorizerのインスタンス
count = CountVectorizer()
#見出しのリストを作ってnumpy配列にする
headline_list = [news[1] for news in train_list]
headline_list.extend([news[1] for news in valid_list])
headline_list.extend([news[1] for news in test_list])
headline_nparr = np.array(headline_list)

#Vectorizerに単語を記憶させる
count.fit_transform(headline_nparr)

#tf-idfを使って単語に重みをつける
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)

#全データセットの特徴量を入れるリスト
feature_list = tfidf.fit_transform(count.fit_transform(headline_nparr)).toarray()

#分割するためのインデックス
train_index = len(train_list)
valid_index = len(train_list) + len(valid_list)
#train, vali, testに分ける
train_feature_list = feature_list[:train_index]
valid_feature_list = feature_list[train_index:valid_index]
test_feature_list = feature_list[valid_index:]

#小数点二桁まで入力できるようにしておく
np.set_printoptions(precision=2)

with open("train.feature.txt", "w") as f:
    for i in range(len(train_list)):
        input = train_list[i][0] + " " + " ".join(str(train_feature_list[i]))
        print(input, file=f)
with open("valid.feature.txt", "w") as f:
    for i in range(len(valid_list)):
        input = valid_list[i][0] + " " + " ".join(str(valid_feature_list[i]))
        print(input, file=f)
with open("test.feature.txt", "w") as f:
    for i in range(len(test_list)):
        input = test_list[i][0] + " " + " ".join(str(test_feature_list[i]))
        print(input, file=f)

"""
akiyama@LAPTOP-8R9KUU89:~/100knocks/第６章$ head -n5 train.feature.txt
e [ 0 .   0 .   0 .   . . .   0 .   0 .   0 . ]
m [ 0 .   0 .   0 .   . . .   0 .   0 .   0 . ]
e [ 0 .   0 .   0 .   . . .   0 .   0 .   0 . ]
b [ 0 .   0 .   0 .   . . .   0 .   0 .   0 . ]
e [ 0 .   0 .   0 .   . . .   0 .   0 .   0 . ]
"""