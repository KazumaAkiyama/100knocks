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

def get_section_level_dict(text):
    section_level_list = re.findall(r"(={2,})(.+?)={2,}", text)
    section_level_dict = {}
    for elem in section_level_list:
        section_level_dict[elem[1]] = len(elem[0]) - 1
    return section_level_dict

text_UK = read_wiki_UK_text()
section_level_dict = get_section_level_dict(text_UK)
print("section : level\n")
for key in section_level_dict.keys():
    print(f"{key} : {section_level_dict[key]}")