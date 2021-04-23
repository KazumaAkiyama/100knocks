from os import name


#ファイルを読み込んで集合に各行を追加していく
with open("popular-names.txt", 'r') as f:
    s = set()
    for line in f:
        s.add(line.split('\t')[0])
#集合の各要素を表示
for elem in s:
    print(elem)