with open("popular-names.txt", 'r') as f:
    s = set()
    for line in f:
        s.add(line.split('\t')[0])

    for elem in s:
        print(elem)