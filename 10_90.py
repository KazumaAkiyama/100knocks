with open("kftt-data-1.0/data/tok/kyoto-train.cln.ja") as f:
    ja_lines = f.readlines()

with open("kftt-data-1.0/data/tok/kyoto-train.cln.en") as f:
    en_lines = f.readlines()

print(ja_lines[0])
print(en_lines[0])

print(ja_lines[1])
print(en_lines[1])

print(ja_lines[2])
print(en_lines[2])

"""
日本 の 水墨 画 を 一変 さ せ た 。

He revolutionized the Japanese ink painting .

諱 は 「 等楊 （ とうよう ） 」 、 もしくは 「 拙宗 （ せっしゅう ） 」 と 号 し た 。

He was given the posthumous name " Toyo " or " Sesshu ( 拙宗 ) . "

備中 国 に 生まれ 、 京都 ・ 相国 寺 に 入 っ て から 周防 国 に 移 る 。

Born in Bicchu Province , he moved to Suo Province after entering SShokoku-ji Temple in Kyoto .
"""