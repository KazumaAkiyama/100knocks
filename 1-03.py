import re

#文字列の作成
str = "Now I need a drink, alcoholic of course," + \
      "after the heavy lectures involving quantum mechanics."

#単語リストの作成
wordList = re.findall('[a-z]+', str, flags=re.IGNORECASE)
#単語ごとの文字数リストを作成
wordCountList = [len(i) for i in wordList]

#文字数リストを表示
print(wordCountList)
