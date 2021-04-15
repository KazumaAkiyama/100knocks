#文字列の作成
patStr = "パトカー"
takStr = "タクシー"

#文字列を１文字ずつ交互に連結
ChimeraStr = ""
for i, j in zip(patStr, takStr): #p.12 2.2
    ChimeraStr += i + j

#文字列の表示
print(patStr)
print(takStr)
print(ChimeraStr)