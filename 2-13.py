#ファイルを開く
col1 = open("col1.txt", 'r')
col2 = open("col2.txt", 'r')
col_marge = open("col_marge.txt", 'w')

#名前と性別をマージして書き込み
for name, sex in zip(col1, col2):
    text = f"{name.strip()}\t{sex}"
    col_marge.write(text)

#ファイルを閉じる
col1.close()
col2.close()
col_marge.close()