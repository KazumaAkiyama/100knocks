#カウント用変数
line_count = 0

#行数もカウント
with open("popular-names.txt", 'r') as f:
    for line in f:
        line_count += 1

#結果の表示
print(line_count)