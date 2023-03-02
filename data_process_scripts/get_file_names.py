with open('clean_valid_labels.txt') as f:
    with open('clean_valid_imgnames.txt', 'w') as o:
        for line in f:
            line = line.split()
            o.write(line[0]+'\n')