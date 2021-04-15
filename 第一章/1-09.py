import random

#受け取った単語の先頭と末尾以外をシャッフル
def ShuffleWord(word):
    return word[0] + "".join(random.sample(word[1:-1], len(word)-2)) + word[-1]

#受け取った単語がn文字より長いかどうか
def IsOverWordCount(word, n):
    if len(word) > n:
        return True
    else:
        return False

#各単語の真ん中をシャッフルした文章を作成する
def GenerateTypoglycemia(str):
    generateStr = []
    for word in str.split():
        if(IsOverWordCount(word, 4)):
            generateStr.append(ShuffleWord(word))
        else:
            generateStr.append(word)
    return " ".join(generateStr)

#文字列の作成
originStr = "I couldn’t believe that I could actually understand what I was reading :" + \
            "the phenomenal power of the human mind ."
#文字列内の単語の中身をシャッフル
typoglycemiaStr = GenerateTypoglycemia(originStr)

#結果の表示
print("元の文章:", originStr)
print("所々入れ替えた文章:", typoglycemiaStr)