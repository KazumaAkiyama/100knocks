from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

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

tsne = TSNE()
tsne.fit(country_vec_list)

plt.figure(figsize=(15,15,), dpi=300)
plt.scatter(tsne.embedding_[:,0], tsne.embeding_[:,1])
for (x,y), country in zip(tsne.embedding_, country_list):
    plt.annotate(country, (x,y))
plt.show()