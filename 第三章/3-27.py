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

def get_basic_info_dict(text):
    basic_info_text = re.findall(r"\{\{基礎情報.*?$(.*?)^\}\}", text, re.MULTILINE + re.DOTALL)
    basic_info_list = re.findall(r"\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))", basic_info_text[0], re.MULTILINE)
    basic_info_dict = {}
    for elem in basic_info_list:
        basic_info_dict[elem[0]] = elem[1]
    return basic_info_dict

def remove_markup(text):
    markup_removed_text = re.sub(r"\'{2,5}","",text)
    return markup_removed_text

def remove_link_markup(text):
    link_markup_removed_text = re.sub(r"\[\[(?:[^:\]]+?\|)?([^:]+?)\]\]", r"\1", text)
    return link_markup_removed_text

text_UK = remove_link_markup(remove_markup(read_wiki_UK_text()))
basic_info_dict = get_basic_info_dict(text_UK)
for key in basic_info_dict.keys():
    print(f"{key} : {basic_info_dict[key]}")