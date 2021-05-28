from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np

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

kmeans = KMeans(n_clusters=5)
kmeans.fit(country_vec_list)
for i in range(5):
    cluster = np.where(kmeans.labels_ == i)[0]
    print('クラス', i)
    print(', '.join([country_list[k] for k in cluster]))

"""
akiyama@LAPTOP-KVF59R5A:~/100knocks/第７章$ python3 7-67.py
クラス 0
Honduras, Ecuador, Peru, Guyana, Venezuela, Philippines, Tuvalu, Nepal, Fiji, Mexico, Chile, Cambodia, Bhutan, Bahamas, Cuba, Jamaica, Belize, Dominica, Nicaragua, Suriname, Laos, Thailand, Samoa, Colombia
クラス 1
Gabon, Tunisia, Somalia, Ghana, Zambia, Nigeria, Kenya, Senegal, Mozambique, Mali, Sudan, Malawi, Niger, Mauritania, Madagascar, Burundi, Eritrea, Namibia, Zimbabwe, Botswana, Rwanda, Angola, Algeria, Guinea, Liberia, Uganda, Gambia
クラス 2
Turkmenistan, Montenegro, Serbia, Lithuania, Slovenia, Latvia, Macedonia, Slovakia, Poland, Belarus, Cyprus, Estonia, Kyrgyzstan, Armenia, Azerbaijan, Russia, Romania, Croatia, Kazakhstan, Georgia, Albania, Tajikistan, Greece, Turkey, Hungary, Bulgaria, Malta, Ukraine, Moldova, Uzbekistan
クラス 3
Morocco, Iraq, Taiwan, Australia, India, Israel, Pakistan, Japan, Malaysia, Bahrain, Iran, Bangladesh, Syria, Egypt, Afghanistan, Korea, Libya, Indonesia, Lebanon, Oman, Jordan, China, Qatar, Vietnam
クラス 4
Europe, Sweden, France, Norway, Austria, USA, Greenland, England, Denmark, Ireland, Switzerland, Iceland, Uruguay, Argentina, Finland, Spain, Netherlands, Italy, Brazil, Canada, Germany, Belgium, Liechtenstein, Portugal
"""