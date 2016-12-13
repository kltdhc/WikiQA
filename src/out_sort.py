import os

def sort(rloc):
    # rloc = '/home/wanghao/workspace/cl/results/dev_2.2455.txt'
    rfile = open(rloc)
    questions = {}
    for line in rfile:
        line = line.strip('\n').split(' ')
        if questions.get(line[0])==None:
            questions[line[0]]=[]
        questions[line[0]].append(float(line[2]))
    rfile.close()

    for q in questions.keys():
        questions[q].sort(reverse=True)

    rfile = open(rloc)
    wfile = open('/home/wanghao/workspace/cl/results/dev_final.txt', 'w')
    for line in rfile:
        line = line.strip('\n').split(' ')
        loc = questions[line[0]].index(float(line[2]))
        wfile.write(line[0]+'\t'+line[1]+'\t'+str(loc+1)+'\n')
        questions[line[0]][loc]=-1
    rfile.close()
    wfile.close()

    os.system('python2 /home/wanghao/workspace/cl/eval.py '+
              '/home/wanghao/workspace/cl/results/dev_final.txt '+
              '/home/wanghao/workspace/cl/data/WikiQA-dev.tsv')

if __name__ == '__main__':
    sort('/home/wanghao/workspace/wikiQA/results/test_3.5819.txt')