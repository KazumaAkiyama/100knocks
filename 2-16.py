import sys

#ファイルの読み込み
names_file = open("popular-names.txt", 'r')
#実行時の引数で分割数を指定する
split_line_num = int(sys.argv[1])
#全部で何行あるか
line_count = 0

#全部の行数と指定された分割数から１ファイルごとの行数を指定する
with open("popular-names.txt", 'r') as f:
    for line in f:
        line_count += 1
file_num = int(line_count / split_line_num)

#1ファイルごとの行数分繰り返す
for i in range(file_num):
    with open("./split_files/names_split" + str(i) + ".txt", 'w') as f:
        #分割数分繰り返す
        for j in range(split_line_num):
            text = names_file.readline()
            f.write(text)

names_file.close()


