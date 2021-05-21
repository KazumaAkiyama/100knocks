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
mat_size = np.unique(test_category_nparr).size
#↑のサイズで&&全部０で初期化した行列
test_mat = np.zeros((mat_size,mat_size), dtype=np.int32)
#testに対して予測したカテゴリーのリスト(数字で入ってる)
test_pre_category_list = lr.predict(test_feature_nparr)
#行は実際のカテゴリ，列は予測したカテゴリ
for x, y in zip(test_category_list, test_pre_category_list):
    test_mat[x, y] += 1

#true-positive　1着予想があたった
tp = test_mat.diagonal()
#false-positive　1着予想がはずれた
fp = test_mat.sum(axis=0) - tp
#false-negative　予想外のやつが1着
fn = test_mat.sum(axis=1) - tp

#適合率の計算
precision = tp / (tp + fp)
#再現率の計算
recall = tp / (tp + fn)
#F1スコアの計算
f1 = 2 * precision * recall / (precision + recall)

#マイクロ平均　全部まとめて平均を計算する
micro_precisiion = tp.sum() / (tp + fn).sum()
micro_recall = tp.sum() / (tp + fp).sum()
micro_f1 = 2 * micro_precisiion * micro_recall / (micro_precisiion + micro_recall)
micro_average = np.array([micro_precisiion, micro_recall, micro_f1])

#マクロ平均　各カテゴリで出した結果の平均
macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
macro_average = np.array([macro_precision, macro_recall, macro_f1])

"""
            再現率　適合率　F１スコア   
        b     ◯◯    ◯◯      ◯◯
        e     ◯◯    ◯◯      ◯◯
        m     ◯◯    ◯◯      ◯◯
        t     ◯◯    ◯◯      ◯◯
マイクロ平均   ◯◯    ◯◯     ◯◯
マクロ平均     ◯◯    ◯◯     ◯◯

ってなる
"""
data_mat = np.array([precision, recall, f1]).T
data_mat = np.vstack([data_mat, micro_average, macro_average])

print(data_mat)

"""
akiyama@LAPTOP-8R9KUU89:~/100knocks/第６章$ python3 6-56.py
[[0.89612676 0.95676692 0.92545455]
 [0.89102564 0.98932384 0.9376054 ]
 [0.95348837 0.46590909 0.6259542 ]
 [0.91089109 0.5974026  0.72156863]
 [0.89670659 0.89670659 0.89670659]
 [0.91288297 0.75235061 0.82487894]]
"""