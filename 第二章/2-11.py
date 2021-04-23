#各行についてタブをスペースに置き換える
with open("popular-names.txt") as f:
    for line in f:
        print(line.replace("\t", " ").replace("\n", ""))

"""
$ sed -e 's/\t/ /g' ./popular-names.txt | head -n 5
Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880
Elizabeth F 1939 1880
Minnie F 1746 1880

"""