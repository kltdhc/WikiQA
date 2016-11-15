rfile = open('/home/wanghao/workspace/wikiQA/result/r1.txt')
questions = {}
for line in rfile:
    line = line.strip('\n').split(' ')
    if questions.get(line[0])==None:
        questions[line[0]]=[]
    questions[line[0]].append(float(line[2]))
rfile.close()

for q in questions.keys():
    questions[q].sort(reverse=True)

rfile = open('/home/wanghao/workspace/wikiQA/result/r1.txt')
wfile = open('/home/wanghao/workspace/cl/r_final.txt', 'w')
for line in rfile:
    line = line.strip('\n').split(' ')
    loc = questions[line[0]].index(float(line[2]))
    wfile.write(line[0]+'\t'+line[1]+'\t'+str(loc+1)+'\n')
    questions[line[0]][loc]=-1
rfile.close()
wfile.close()