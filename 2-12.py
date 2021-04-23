#書き込み用のファイルを開く
col1 = open("col1.txt", 'w')
col2 = open("col2.txt", 'w')

#タブ区切りで一列目と二列目をそれぞれ書き込み
with open("popular-names.txt", 'r') as f:
    for line in f:
        col1.write(line.split('\t')[0] + '\n')
        col2.write(line.split('\t')[1] + '\n')

#ファイルを閉じる
col1.close()
col2.close()