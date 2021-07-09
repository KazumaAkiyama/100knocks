#train.tsvの作成
with open("train.sub.ja") as f:
    ja_lines = f.readlines()

with open("train.sub.en") as f:
    en_lines = f.readlines()

with open("train_sub.tsv", "w") as f:
    for i in range(len(ja_lines)):
        text = ja_lines[i].replace("\n", "") + "\t" + en_lines[i]
        f.write(text)

#valid.tsvの作成
with open("dev.sub.ja") as f:
    ja_lines = f.readlines()

with open("dev.sub.en") as f:
    en_lines = f.readlines()

with open("valid_sub.tsv", "w") as f:
    for i in range(len(ja_lines)):
        text = ja_lines[i].replace("\n", "") + "\t" + en_lines[i]
        f.write(text)

#test.tsvの作成
with open("test.sub.ja") as f:
    ja_lines = f.readlines()

with open("test.sub.en") as f:
    en_lines = f.readlines()

with open("test_sub.tsv", "w") as f:
    for i in range(len(ja_lines)):
        text = ja_lines[i].replace("\n", "") + "\t" + en_lines[i]
        f.write(text)