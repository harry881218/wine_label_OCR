import csv

# with open('test.txt', 'r') as f:
#     with open('csvformat.txt', 'w') as out_file:
#         # writer = csv.writer(out_file)
#         # writer.writerow(('filename', 'words'))
#         for line in f:
#             l = line.split()
#             name = l[0]
#             l.pop(0)
#             label = ' '.join(l)
#             output = name + "," + label + '\n'
#             out_file.write(output)
#             #e = (name, label)
#             # print(e)
#             # writer.writerows(e)


with open('csvformat.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('test.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('filename', 'words'))
        writer.writerows(lines)