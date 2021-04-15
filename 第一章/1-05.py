#受け取ったlistのn文字ずつのn-gramを返す
def n_gram(list, n):
    return [list[i:i+n] for i in range(len(list)-(n-1))]

#文字列を作成
str = "I am an NLPer"
#単語リストを作成
wordList = str.split()

#n-gramを表示
print(n_gram(wordList, 2))
print(n_gram(str, 2))