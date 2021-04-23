with open("popular-names.txt", "r") as f:
    line_list = f.readlines()
    line_list.sort(key=lambda line: line.split('\t')[2], reverse=True)
    for line in line_list:
        print(line)