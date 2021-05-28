from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

country_list = []

with open("questions-words.txt") as f:
    for line in f:
        word_list = line.split()
        if word_list[0] == ":":
            label = word_list[1]
        elif label in ["capital-common-countries", "capital-world"]:
            country_list.append(word_list[1])
        elif label in ["currency", "gram6-nationality-adjective"]:
            country_list.append(word_list[0])

country_list = list(set(country_list))

country_vec_list = [model[country] for country in country_list]

plt.figure(figsize=(16, 9), dpi=200)
Z = linkage(country_vec_list, method='ward')
dendrogram(Z, labels=country_list)
plt.show()