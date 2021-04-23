import sys

names_file = open("popular-names.txt", 'r')
split_line_num = int(sys.argv[1])
line_count = 0
with open("popular-names.txt", 'r') as f:
    for line in f:
        line_count += 1
file_num = int(line_count / split_line_num)

for i in range(file_num):
    with open("./split_files/names_split" + str(i) + ".txt", 'w') as f:
        for j in range(split_line_num):
            text = names_file.readline()
            f.write(text)


