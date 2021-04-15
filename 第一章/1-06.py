#受け取ったlistのn文字ずつのn-gramを返す
def n_gram(list, n):
    return [list[i:i+n] for i in range(len(list)-(n-1))]

#文字列の作成
str1 = "paraparaparadise"
str2 = "paragraph"

#n-gramを作成し集合にする
X = set(n_gram(str1, 2))
Y = set(n_gram(str2, 2))

#結果を表示
print("Xの集合:", X)
print("Yの集合:", Y)
print("XとYの和集合:", X | Y)
print("XとYの積集合:", X & Y)
print("XとYの差集合:", X - Y)
print("Xにseは含まれて", "いる" if "se" in X else "いない")
print("Yにseは含まれて", "いる" if "se" in Y  else "いない")