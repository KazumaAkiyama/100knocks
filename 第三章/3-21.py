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

def get_category_line_list(text):
    category_line_list = re.findall(r"\[\[Category:.*\]\]", text, re.MULTILINE)
    return category_line_list

text_UK = read_wiki_UK_text()
category_line_list = get_category_line_list(text_UK)
for line in category_line_list:
    print(line)

"""
akiyama@LAPTOP-8R9KUU89:~/100knocks/第三章$ python3 3-21.py
[[Category:イギリス|*]]
[[Category:イギリス連邦加盟国]]
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国|元]]
[[Category:海洋国家]]
[[Category:現存する君主国]]
[[Category:島国]]
[[Category:1801年に成立した国家・領域]]
"""