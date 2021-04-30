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

def get_mediafile_list(text):
    mediafile_list = re.findall(r"\[\[ファイル:(.+?)\|", text)
    return mediafile_list

text_UK = read_wiki_UK_text()
mediafile_list = get_mediafile_list(text_UK)
for i in mediafile_list:
    print(i)