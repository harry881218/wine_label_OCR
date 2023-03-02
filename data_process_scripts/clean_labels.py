with open('bbox_labels.txt') as f:
    with open('clean_bbox_labels.txt', 'w') as o:
        for line in f:
            #print("{} has len {}".format(line, len(line)))
            if len(line.split()) > 1:
                #print(line)
                o.write(line)