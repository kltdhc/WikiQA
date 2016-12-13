import os
import numpy as np
from model import train_model
import cnn_model


def w2v_input(file_dir):
    # file not exist
    if file_dir == None or not os.path.isfile(file_dir):
        print(file_dir + ' is not exist')
        return None
    file = open(file_dir, 'r')
    out = {}
    for line in file:
        line = line.strip('\n').split(' ')
        word = line[0]
        vec = [float(_s) for _s in line[1:]]
        out[word] = vec
    file.close()
    return out


def sens_input(file_dir):
    # file not exist
    ret = 0
    if file_dir == None or not os.path.isfile(file_dir):
        print(file_dir + ' is not exist')
        ret = 1
    infile = open(file_dir, 'r')
    sentences = []
    sen_no = {}
    i = 0
    for line in infile:
        line = line.strip('\n')
        line = line.split('\t')
        if len(line) < 2:
            continue
        sentences.append(line[1])
        sen_no[line[0]] = len(sentences)-1
    infile.close()
    return sen_no, sentences


class VocabularyProcessor:
    def __init__(self, dim, word_dict={'': 0}):
        # word_dict[' ']=0 should be contained in word_dict
        self.vocab = word_dict
        self.dim = dim
        self.wordcount = len(word_dict)
        self.wordorder = {}
        for i in word_dict:
            self.wordorder[word_dict.get(i)] = i

    def getwords(self):
        return list(self.wordorder.values())

    def sentencepreprocessing(self, sentences):
        # if sentences contains whose length > dim, the return value will be None
        ret = []
        lenwords = []
        for sen in sentences:
            words = sen.split(' ')
            vec = [0 for _ in range(self.dim)]
            if (len(words) > self.dim):
                return None
            for i in range(len(words)):
                if self.vocab.get(words[i]) == None:
                    self.vocab[words[i]] = self.wordcount
                    self.wordorder[self.wordcount] = words[i]
                    self.wordcount += 1
                vec[i] = self.vocab.get(words[i])
            ret.append(vec)
            lenwords.append(len(words))
        return ret, lenwords


def w2v_init(words, w2v):
    ret = []
    dim = len(w2v['this'])
    for i in words:
        i = i.lower()
        if w2v.get(i) == None:
            ret.append([0 for _ in range(dim)])
        else:
            ret.append(w2v[i])
    return ret


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def sentence_group_input(right_file, wrong_file, sen_no):
    sen_q=[]
    sen_ar=[]
    sen_aw=[]
    right_ans={}
    with open(right_file) as rightin:
        for line in rightin:
            line = line.strip('\n').split('\t')
            if len(line) < 2:
                continue
            if right_ans.get(line[0])==None:
                right_ans[line[0]] = [line[1]]
            else:
                right_ans[line[0]].append(line[1])
    with open(wrong_file) as wrongin:
        for line in wrongin:
            line = line.strip('\n').split('\t')
            if len(line) < 2:
                continue
            if right_ans.get(line[0]) == None:
                continue
            for i in right_ans[line[0]]:
                if sen_no[line[0]]==None or sen_no[i]==None or sen_no[line[1]]==None:
                    continue
                sen_q.append(sen_no[line[0]])
                sen_ar.append(sen_no[i])
                sen_aw.append(sen_no[line[1]])
    return sen_q, sen_ar, sen_aw

def read_test(file_name, sen_no):
    rfile = open(file_name)
    outq = []
    outa = []
    inq = []
    ina = []
    for line in rfile:
        line = line.strip('\n').split('\t')
        if sen_no.get(line[0])!=None and sen_no.get(line[1])!=None:
            outq.append(sen_no[line[0]])
            outa.append(sen_no[line[1]])
            inq.append(line[0])
            ina.append(line[1])
    return outq, outa, inq, ina

if __name__ == '__main__':
    (sen_no, sentences) = sens_input('/home/wanghao/workspace/cl/wikiqa/lemma_sen.txt')
    print(len(sentences))
    # print('A: ' + sen1[0], '\nB: ' + sen2[0], '\nO:', out[0])
    # print(sen_no['Q1'], sentences[sen_no['Q1']], len(sentences))
    max_document_length = max([len(x.split(" ")) for x in sentences])
    print(max_document_length)
    (sen_q, sen_ar, sen_aw) = sentence_group_input(
        '/home/wanghao/workspace/cl/train_right.txt',
        '/home/wanghao/workspace/cl/train_wrong.txt',
        sen_no
    )
    print(len(sen_q))
    vocab = VocabularyProcessor(max_document_length)
    senmat, lenwords = vocab.sentencepreprocessing(sentences)
    words = vocab.getwords()
    # print(sen1mat[1])
    # print(sen2mat[1])
    for i in senmat[1]:
        if i == 0:
            break
    # print(words[i], end=" ")
    # print()
    #    print(words[i], end=" ")
    # w2v = w2v_input('/media/wanghao/0DD813960DD81396/work/summer_try/glove.6B/glove.6B.100d.txt')
    w2v = w2v_input("/home/wanghao/workspace/cl/w2v.txt")
    # print(len(w2v))
    # print(w2v['this'])
    wordmat = w2v_init(words, w2v)
    print('Read Embedding Mat: Done!')
    # print(w2v[words[senmat[2][4]]])
    # print(wordmat[senmat[2][4]])
    testq, testa, intestq, intesta = read_test('/home/wanghao/workspace/cl/dev.txt', sen_no)
    test2q, test2a, intest2q, intest2a = read_test('/home/wanghao/workspace/cl/test.txt', sen_no)
    print(len(testq))
    t = train_model(max_document_length,
                    wordmat,
                    sen_q,
                    sen_ar,
                    sen_aw,
                    senmat,
                    lenwords,
                    testq,
                    testa,
                    intestq,
                    intesta,
                    test2q,
                    test2a,
                    intest2q,
                    intest2a)

    # print(sen_q[0], sentences[sen_q[0]])
    # print(sen_ar[0], sentences[sen_ar[0]])
    # print(sen_aw[0], sentences[sen_aw[0]])
