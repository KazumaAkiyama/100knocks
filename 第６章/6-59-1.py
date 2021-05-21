import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import japanize_matplotlib

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

#正則化をいっぱい用意する
parameter_list = np.arange(0.1, 5.1, 0.1)
#モデルもいっぱい用意する
lr_list = [LogisticRegression(tol=parameter, max_iter=1000).fit(train_feature_nparr, train_category_nparr) for parameter in parameter_list]

#正解率計算のための関数
def accuracy(lr, feature_nparr, category_nparr):
    correct_count = 0
    pre_category_list = lr.predict(feature_nparr)
    for i in range(len(feature_nparr)):
        if pre_category_list[i] == category_nparr[i]:
            correct_count += 1
    accuracy = float(correct_count / len(feature_nparr))
    return accuracy

#正解率の細かいリストができる
train_accs = [accuracy(lr, train_feature_nparr, train_category_nparr) for lr in lr_list]
valid_accs = [accuracy(lr, valid_feature_nparr, valid_category_nparr) for lr in lr_list]
test_accs = [accuracy(lr, test_feature_nparr, test_category_nparr) for lr in lr_list]

plt.plot(parameter_list, train_accs, label = '学習')
plt.plot(parameter_list, valid_accs, label = '検証')
plt.plot(parameter_list, test_accs, label = '評価')
plt.legend()
plt.show()