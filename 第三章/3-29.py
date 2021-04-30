import requests
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

text_UK = read_wiki_UK_text()
basic_info_dict = get_basic_info_dict(text_UK)
file_name = basic_info_dict["国旗画像"].replace(" ", "_")
url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + file_name + '&prop=imageinfo&iiprop=url&format=json'
file_data = requests.get(url)
print(re.search(r'"url":"(.+?)"', file_data.text).group(1))
