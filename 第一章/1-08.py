#受け取った文字列の小文字を変換する
def cipher(str):
    crpStr = []
    for w in str:
        crpWord = chr(219-ord(w)) if w.islower() else w
        crpStr.append(crpWord)
    return "".join(crpStr)

#文字列の作成
message = "I am chicken crisp."
#文字列の暗号化
encrpMsg = cipher(message)
#文字列の復号化
decrpMsg = cipher(encrpMsg)

#結果の表示
print("元のメッセージ:", message)
print('暗号化したメッセージ:', encrpMsg)
print('復号化したメッセージ:', decrpMsg)