#辞書を作成
name_freq_dict = {}

#辞書の一致する項目があればインクリメント
with open("popular-names.txt", "r") as f:
    for line in f:
        key = line.split('\t')[0]
        name_freq_dict[key] = name_freq_dict.get(key, 0) + 1

#それをつかってソート
    for name in sorted(name_freq_dict.items(), key = lambda fre : fre[1], reverse = True):
        print(name[0])