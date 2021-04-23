#各行についてタブをスペースに置き換える
with open("popular-names.txt") as f:
    for line in f:
        print(line.replace("\t", " ").replace("\n", ""))