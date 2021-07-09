#train.tsvの作成
with open("kftt-data-1.0/data/tok/kyoto-train.cln.ja") as f:
    ja_lines = f.readlines()

with open("kftt-data-1.0/data/tok/kyoto-train.cln.en") as f:
    en_lines = f.readlines()

with open("train.tsv", "w") as f:
    for i in range(len(ja_lines)):
        text = ja_lines[i].replace("\n", "") + "\t" + en_lines[i]
        f.write(text)

#valid.tsvの作成
with open("kftt-data-1.0/data/tok/kyoto-dev.ja") as f:
    ja_lines = f.readlines()

with open("kftt-data-1.0/data/tok/kyoto-dev.en") as f:
    en_lines = f.readlines()

with open("valid.tsv", "w") as f:
    for i in range(len(ja_lines)):
        text = ja_lines[i].replace("\n", "") + "\t" + en_lines[i]
        f.write(text)

#test.tsvの作成
with open("kftt-data-1.0/data/tok/kyoto-test.ja") as f:
    ja_lines = f.readlines()

with open("kftt-data-1.0/data/tok/kyoto-test.en") as f:
    en_lines = f.readlines()

with open("test.tsv", "w") as f:
    for i in range(len(ja_lines)):
        text = ja_lines[i].replace("\n", "") + "\t" + en_lines[i]
        f.write(text)