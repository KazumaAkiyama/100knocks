import json
import re

def read_wiki_UK_text():
    with open("jawiki-country.json") as f:
        for line in f:
            country_dict = json.loads(line)
            if country_dict["title"] == "イギリス":
                text_UK = country_dict["text"]
                break
    return text_UK

def get_category_name_list(text):
    category_name_list = re.findall(r"\[\[Category:(.*?)(?:\|.*)?\]\]", text, re.MULTILINE)
    return category_name_list

text_UK = read_wiki_UK_text()
category_name_list = get_category_name_list(text_UK)
for line in category_name_list:
    print(line)

"""
akiyama@LAPTOP-8R9KUU89:~/100knocks/第三章$ python3 3-22.py
イギリス
イギリス連邦加盟国
英連邦王国
G8加盟国
欧州連合加盟国
海洋国家
現存する君主国
島国
1801年に成立した国家・領域
"""