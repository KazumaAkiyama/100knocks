import re

#文字列の作成
str = "Hi He Lied Because Boron Could Not Oxidize Fluorine." + \
      "New Nations Might Also Sign Peace Security Clause. Arthur King Can."
#単語リストの作成
wordList = re.findall('[a-z]+', str, flags=re.IGNORECASE)
#何番目の単語の1文字目を抽出するかのリスト
extract1stCharIndex = [1,5,6,7,8,9,15,16]
#辞書型リストの作成　
# p.16 2.3 を使いたいけどなんの辞書かわかってないときどうすればいいんですかね？
dictionaly = {}

#単語リストの先頭1文字or2文字を"何番目の単語か"という情報と一緒に格納
for i, word in enumerate(wordList, 1):
    if i in extract1stCharIndex:
        dictionaly[word[:1:]] = i
    else:
        dictionaly[word[:2:]] = i

#辞書型リストを表示
print(dictionaly)
