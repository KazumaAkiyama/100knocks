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

#trainに対する正解率の計算　合ってる文数えて割り算してるだけ
correct_count = 0
train_pre_category_list = lr.predict(train_feature_nparr)
for i in range(len(train_feature_nparr)):
    if train_pre_category_list[i] == train_category_list[i]:
        correct_count += 1
train_accuracy = float(correct_count / len(train_feature_nparr))
print(train_accuracy)

#testに対する正解率の計算　上に同じく　
correct_count = 0
test_pre_category_list = lr.predict(test_feature_nparr)
for i in range(len(test_feature_nparr)):
    if test_pre_category_list[i] == test_category_list[i]:
        correct_count += 1
test_accuracy = float(correct_count / len(test_feature_nparr))
print(test_accuracy)

"""
akiyama@LAPTOP-8R9KUU89:~/100knocks/第６章$ python3 6-54.py
0.9457132160239611
0.8967065868263473
"""