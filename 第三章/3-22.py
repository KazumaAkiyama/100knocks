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