#意味的な部分のラベル名
sem_list = {
    "capital-common-countries",
    "capital-world",
    "currency",
    "city-in-state",
    "family"
}
#文法的な部分のラベル名
syn_list = {
    "gram1-adjective-to-adverb",
	"gram2-opposite",
	"gram3-comparative",
	"gram4-superlative",
	"gram5-present-participle",
	"gram6-nationality-adjective",
	"gram7-past-tense",
	"gram8-plural",
	"gram9-plural-verbs"
}

#単語数と正解数のカウント
sem_word_count = 0
sem_correct_count = 0
syn_word_count = 0
syn_correct_count = 0


with open("question-words-similarity.txt") as f:
    for line in f:
        word_list = line.split()
        #ラベル名更新
        if word_list[0] == ":":
            label = word_list[1]
        #ラベル名ごとにインクリメント
        elif label in sem_list:
            sem_word_count += 1
            if word_list[3] == word_list[4]:
                sem_correct_count += 1
        elif label in syn_list:
            syn_word_count += 1
            if word_list[3] == word_list[4]:
                syn_correct_count += 1

#正解率の計算
sem_accuracy = sem_correct_count/sem_word_count
syn_accuracy = syn_correct_count/syn_word_count

print("意味的アナロジー正解率：", sem_accuracy)
print("文法的アナロジー正解率", syn_accuracy)

"""
意味的アナロジー正解率: 0.7308602999210734
文法的アナロジー正解率: 0.7400468384074942
"""
