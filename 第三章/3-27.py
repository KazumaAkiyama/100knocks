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

"""
akiyama@LAPTOP-8R9KUU89:~/100knocks/第三章$ python3 3-27.py
略名 : イギリス
日本語国名 : グレートブリテン及び北アイルランド連合王国
国旗画像 : Flag of the United Kingdom.svg
国章画像 : [[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]
国章リンク : （国章）
標語 : {{lang|fr|Dieu et mon droit}}<br />（フランス語:神と我が権利）
国歌 : {{lang|en|God Save the Queen}}{{en icon}}<br />神よ女王を護り賜え<br />{{center|[[ファイ ル:United States Navy Band - God Save the Queen.ogg]]}}
地図画像 : Europe-UK.svg
位置画像 : United Kingdom (+overseas territories) in the World (+Antarctica claims).svg
公用語 : 英語
首都 : ロンドン（事実上）
最大都市 : ロンドン
元首等肩書 : 女王
元首等氏名 : エリザベス2世
首相等肩書 : 首相
首相等氏名 : ボリス・ジョンソン
他元首等肩書1 : 貴族院議長
他元首等氏名1 : [[:en:Norman Fowler, Baron Fowler|ノーマン・ファウラー]]
他元首等肩書2 : 庶民院議長
他元首等氏名2 : {{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}
他元首等肩書3 : 最高裁判所長官
他元首等氏名3 : [[:en:Brenda Hale, Baroness Hale of Richmond|ブレンダ・ヘイル]]
面積順位 : 76
面積大きさ : 1 E11
面積値 : 244,820
水面積率 : 1.3%
人口統計年 : 2018
人口順位 : 22
人口大きさ : 1 E7
人口値 : 6643万5600<ref>{{Cite web|url=https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates|title=Population estimates - Office for National Statistics|accessdate=2019-06-26|date=2019-06-26}}</ref>
人口密度値 : 271
GDP統計年元 : 2012
GDP値元 : 1兆5478億<ref name="imf-statistics-gdp">[http://www.imf.org/external/pubs/ft/weo/2012/02/weodata/weorept.aspx?pr.x=70&pr.y=13&sy=2010&ey=2012&scsm=1&ssd=1&sort=country&ds=.&br=1&c=112&s=NGDP%2CNGDPD%2CPPPGDP%2CPPPPC&grp=0&a=IMF>Data and Statistics>World Economic Outlook Databases>By Countrise>United Kingdom]</ref>
GDP統計年MER : 2012
GDP順位MER : 6
GDP値MER : 2兆4337億<ref name="imf-statistics-gdp" />
GDP統計年 : 2012
GDP順位 : 6
GDP値 : 2兆3162億<ref name="imf-statistics-gdp" />
GDP/人 : 36,727<ref name="imf-statistics-gdp" />
建国形態 : 建国
確立形態1 : イングランド王国／スコットランド王国<br />（両国とも1707年合同法まで）
確立年月日1 : 927年／843年
確立形態2 : グレートブリテン王国成立<br />（1707年合同法）
確立年月日2 : 1707年{{0}}5月{{0}}1日
確立形態3 : グレートブリテン及びアイルランド連合王国成立<br />（1800年合同法）
確立年月日3 : 1801年{{0}}1月{{0}}1日
確立形態4 : 現在の国号「グレートブリテン及び北アイルランド連合王国」に変更
確立年月日4 : 1927年{{0}}4月12日
通貨 : UKポンド (£)
通貨コード : GBP
時間帯 : ±0
夏時間 : +1
ISO 3166-1 : GB / GBR
ccTLD : .uk / .gb<ref>使用は.ukに比べ圧倒的少数。</ref>
国際電話番号 : 44
注記 : <references/>
"""