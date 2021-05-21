import numpy as np
from sklearn.linear_model import LogisticRegression
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

#カテゴリーをリストにして0-4の数字に対応させる
categorie_list = ['b', 'e', 'm', 't']
#カテゴリーの文字を数字に置き換える
train_category_list = [categorie_list.index(category) for category, _ in train_list]
valid_category_list = [categorie_list.index(category) for category, _ in valid_list]
test_category_list = [categorie_list.index(category) for category, _ in test_list]

#全部numpyに変換する
train_category_nparr = np.array(train_category_list, dtype=np.int8)
valid_category_nparr = np.array(valid_category_list, dtype=np.int8)
test_category_nparr = np.array(test_category_list, dtype=np.int8)
train_feature_nparr = np.array(train_feature_list, dtype=np.float32)
valid_feature_nparr = np.array(valid_feature_list, dtype=np.float32)
test_feature_nparr = np.array(test_feature_list, dtype=np.float32)

#なんかよくわからないけどこれで学習できるらしい
lr = LogisticRegression(max_iter=1000)
lr.fit(train_feature_nparr, train_category_nparr)

#行列のサイズを指定してる　一応カテゴリ数準拠
mat_size = np.unique(train_category_nparr).size

#↑のサイズで&&全部０で初期化した行列 
train_mat = np.zeros((mat_size,mat_size), dtype=np.int32)
#trainに対して予測したカテゴリーのリスト(数字で入ってる)
train_pre_category_list = lr.predict(train_feature_nparr)
#行は実際のカテゴリ，列は予測したカテゴリ
for x, y in zip(train_category_list, train_pre_category_list):
    train_mat[x, y] += 1
print(train_mat)

#testに対してやってるだけ
test_mat = np.zeros((mat_size,mat_size), dtype=np.int32)
test_pre_category_list = lr.predict(test_feature_nparr)
for x, y in zip(test_category_list, test_pre_category_list):
    test_mat[x, y] += 1
print(test_mat)

"""
akiyama@LAPTOP-8R9KUU89:~/100knocks/第６章$ python3 6-55.py
[[4417   56    4   34]
 [  18 4196    0    5]
 [  79  123  522    2]
 [ 147  110    2  969]]
[[509  17   0   6]
 [  5 556   0   1]
 [ 22  23  41   2]
 [ 32  28   2  92]]
"""
