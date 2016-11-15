import csv
sentences={}

with open('/home/wanghao/workspace/cl/data/WikiQA-train.tsv') as inputfile:
    reader = csv.reader(inputfile, delimiter='\t')
    writer = open('/home/wanghao/workspace/cl/train_wrong.txt', 'w')
    for row in reader:
        if(row[6]=='0'):
            writer.write(row[0]+'\t'+row[4]+'\n')
    writer.close()
