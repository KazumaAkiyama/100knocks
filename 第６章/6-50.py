import random

#ファイルを開いて一行ごとにリストに格納
with open("newsCorpora.csv") as f:
    news_list = f.readlines()

#各行をタブ区切りの要素に分けて格納
news_list = [line.split("\t") for line in news_list]

#採用する出版社のリスト
publisher_list = [
    "Reuters",
    "Huffington Post",
    "Businessweek",
    "Contactmusic.com",
    "Daily Mail",
]

#該当する出版社のニュースを格納
news_list = [news for news in news_list if news[3] in publisher_list]

#カテゴリと見出しだけ格納
news_list = [[news[4], news[1]] for news in news_list]

#ランダムに並び替える
random.shuffle(news_list)

#分割するための要素番号を作成
train_index = int(len(news_list)*0.8)
valid_index = int(len(news_list)*0.9)

#train, vaild, testに分ける
train_list = news_list[:train_index]
valid_list = news_list[train_index:valid_index]
test_list = news_list[valid_index:]

#ファイルに書き込んで保存する
with open("train.txt", "w") as f:
    for news in train_list:
        print("\t".join(news), file=f)
with open("valid.txt", "w") as f:
    for news in valid_list:
        print("\t".join(news), file=f)
with open("test.txt", "w") as f:
    for news in test_list:
        print("\t".join(news), file=f)

#渡したリストのカテゴリの事例数を数えて表示する
def display_category_count(news_list):
    b_count = 0
    e_count = 0
    m_count = 0
    t_count = 0
    for news in news_list:
        if news[0] == "b":
            b_count += 1
        if news[0] == "e":
            e_count += 1
        if news[0] == "m":
            m_count += 1
        if news[0] == "t":
            t_count += 1
    print("b:" + str(b_count))
    print("e:" + str(e_count))
    print("m:" + str(m_count))
    print("t:" + str(t_count))

print("<<trainのカテゴリの事例数>>")
display_category_count(train_list)
print("\n<<validのカテゴリの事例数>>")
display_category_count(valid_list)
print("\n<<testのカテゴリの事例数>>")
display_category_count(test_list)

"""
<<trainのカテゴリの事例数>>
b:4511
e:4219
m:726
t:1228

<<validのカテゴリの事例数>>
b:584
e:513
m:96
t:143

<<testのカテゴリの事例数>>
b:532
e:562
m:88
t:154
"""
