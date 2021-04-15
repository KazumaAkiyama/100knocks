#受け取った3変数をテンプレ文に当てはめて返す
def GenerateSentence(x, y, z):
    return "{when}時の{subject}は{status}".format(when = x, subject = y, status = z)

#結果を表示
print(GenerateSentence(12, "気温", 22.4))