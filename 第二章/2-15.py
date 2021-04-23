import sys

#実行時に引数を受け取る
read_taillines_num = int(sys.argv[1])

#ファイルから行ごとのリストを作成
with open("popular-names.txt", 'r') as f:
    line_list = f.readlines()

#受け取った引数まで表示
print("".join(line_list[-read_taillines_num:]))
