#ファイルを読み込んでリストにする
with open("popular-names.txt", "r") as f:
    line_list = f.readlines()
    #タブ区切りの3行目についてソート
    line_list.sort(key=lambda line: line.split('\t')[2], reverse=True)
    #リストの各要素を表示
    for line in line_list:
        print(line)