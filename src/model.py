import lstm
import tensorflow as tf
import numpy as np
import out_sort

class train_model:
    def __init__(self, senlen, wordmat, sen_q, sen_ar,
                 sen_aw, sentences, senlens, testq, testa,
                 intestq, intesta, test2q, test2a, intest2q, intest2a):
        # output length of lstm cells
        n_hidden = 100
        outputlen = n_hidden * 2
        batch_size = 36
        learning_rate = 5e-3
        training_iter = 80000
        margin = 0.2
        # highest average allowed
        min_allow = 0.05
        # execute test set or not
        test = 0
        alpha = 1e-3
        embedding_size = len(wordmat[0])
        # shuffle data
        senq = np.array(sen_q)
        senar = np.array(sen_ar)
        senaw = np.array(sen_aw)
        sent = np.array(sentences)
        tsentq = np.array(testq)
        tsenta = np.array(testa)
        lenwords = np.array(senlens)
        tsent2q = np.array(test2q)
        tsent2a = np.array(test2a)
        np.random.seed(100)
        shuffle_indices = np.random.permutation(np.arange(len(senq)))
        senq_train = senq[shuffle_indices]
        senar_train = senar[shuffle_indices]
        senaw_train = senaw[shuffle_indices]
        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                # basic cells
                self.wordmat = tf.Variable(wordmat, name='wordmat')
                self.embedding()
                self.ques_lstm = lstm.SenLSTM(
                    self.lstm_in_ques,
                    embedding_size, senlen,outputlen,
                    self.wordmat, n_hidden, 2, batch_size, 'queslstm'
                )

                self.ans_lstm = lstm.SenLSTM(
                    self.lstm_in_ans,
                    embedding_size, senlen, outputlen,
                    self.wordmat, n_hidden, 2, batch_size*2, 'anslstm'
                )
                step = 0

                # similarity of the vectors
                ans1 = tf.slice(self.ans_lstm.lstm_out, [0, 0], [batch_size, outputlen])
                ans2 = tf.slice(self.ans_lstm.lstm_out, [batch_size, 0], [batch_size, outputlen])
                ques = self.ques_lstm.lstm_out

                W = tf.Variable(tf.truncated_normal([outputlen, outputlen], stddev=0.1))
                q_mul = tf.matmul(ques, W)
                qa1_cos = tf.reshape(tf.reduce_sum(tf.mul(q_mul, ans1), reduction_indices=1), [-1,1])
                qa2_cos = tf.reshape(tf.reduce_sum(tf.mul(q_mul, ans2), reduction_indices=1), [-1,1])

                # cosine similarity is also available
                # qa1_cos = tf.nn.tanh(qa1_cos, name='qa1_score')
                # qa2_cos = tf.nn.tanh(qa2_cos, name='qa2_score')

                # W = tf.Variable(tf.truncated_normal([outputlen * 2, 1], stddev=0.1))
                # qa1_cos = tf.matmul(tf.concat(1, [ques, ans1]), W)
                # qa2_cos = tf.matmul(tf.concat(1, [ques, ans2]), W)


                # qa1 = tf.matmul(tf.mul(ques, ans1), tf.ones([outputlen, 1], tf.float32))
                # qa2 = tf.matmul(tf.mul(ques, ans2), tf.ones([outputlen, 1], tf.float32))
                # a1sq = tf.matmul(tf.sqrt(tf.mul(ans1, ans1)), tf.ones([outputlen, 1], tf.float32))
                # a2sq = tf.matmul(tf.sqrt(tf.mul(ans2, ans2)), tf.ones([outputlen, 1], tf.float32))
                # qsq = tf.matmul(tf.sqrt(tf.mul(ques, ques)), tf.ones([outputlen, 1], tf.float32))
                # qa1_cos = tf.div(qa1, tf.mul(a1sq, qsq))
                # qa2_cos = tf.div(qa2, tf.mul(a2sq, qsq))

                loss = tf.matmul(tf.ones([1, batch_size], tf.float32),
                                 tf.maximum(tf.zeros([batch_size, 1], tf.float32),
                                            margin * tf.ones([batch_size, 1], tf.float32) + qa2_cos - qa1_cos))
                loss_all = loss + alpha * tf.nn.l2_loss(W)
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss_all)
                pred = tf.argmax(tf.concat(1, [qa1_cos, qa2_cos]),1)
                accuracy = tf.reduce_mean(tf.cast(
                        tf.equal(pred, tf.zeros([batch_size, 1], tf.int64)),
                        tf.float32))
                step = 1
                init = tf.initialize_all_variables()
                sess.run(init)
                print('Model Defination: Done!')
                mincost=999
                while step * batch_size < training_iter:
                    ques_in, ans_in, lenques_in, lenans_in = self.get_batch(
                        sent, lenwords, senq_train,
                        senar_train, senaw_train, step, batch_size
                    )
                    _, acc, cost, cost_all = sess.run([optimizer, accuracy, loss, loss_all], feed_dict={
                                       self.input_ques:ques_in,
                                       self.input_ans:ans_in,
                                       self.ques_lstm.input_len: lenques_in,
                                       self.ans_lstm.input_len: lenans_in,
                                       self.ques_lstm.keep_prob: 0.8,
                                       self.ans_lstm.keep_prob: 0.8
                                   })

                    if cost < min_allow * batch_size and step * batch_size >= 4000:
                        if mincost > cost_all:
                            mincost = float('%.4f' % cost_all)
                        teststep = 0
                        self.result = []
                        print('Cost_Average = %.4f, Cost_all = %.4f, Begin test...'%(cost/batch_size, cost_all))
                        while teststep * batch_size < tsentq.shape[0]:
                            ques_in = sent[tsentq[teststep * batch_size: (teststep + 1) * batch_size]]
                            ans_in = sent[tsenta[teststep * batch_size: (teststep + 1) * batch_size]]
                            lenques_in = lenwords[tsentq[teststep * batch_size: (teststep + 1) * batch_size]]
                            lenans_in = lenwords[tsenta[teststep * batch_size: (teststep + 1) * batch_size]]
                            leng_ques = ques_in.shape[0]
                            if (leng_ques < batch_size):
                                i = np.array(
                                    [[0 for _i in range(0, senlen)] for _j in range(leng_ques, batch_size)])
                                j = np.array([1 for _j in range(leng_ques, batch_size)])
                                ques_in = np.concatenate([ques_in, i], axis=0)
                                ans_in = np.concatenate([ans_in, i], axis=0)
                                lenques_in = np.concatenate([lenques_in, j], axis=0)
                                lenans_in = np.concatenate([lenans_in, j], axis=0)
                            ans_in = np.concatenate((ans_in, ans_in), axis=0)
                            lenans_in = np.concatenate((lenans_in, lenans_in), axis=0)
                            thisr = sess.run(qa1_cos, feed_dict={
                                self.input_ques: ques_in,
                                self.input_ans: ans_in,
                                self.ques_lstm.input_len: lenques_in,
                                self.ans_lstm.input_len: lenans_in,
                                self.ques_lstm.keep_prob: 1.0,
                                self.ans_lstm.keep_prob: 1.0
                            })
                            teststep = teststep + 1
                            self.result += thisr.tolist()
                        outf = open('/home/wanghao/workspace/wikiQA/results/dev_%.4f.txt'%cost_all, 'w')
                        for i in range(len(intestq)):
                            print(intestq[i], intesta[i], self.result[i][0], file=outf)
                        outf.close()
                        out_sort.sort('/home/wanghao/workspace/wikiQA/results/dev_%.4f.txt'%cost_all)
                        if test==1:
                            teststep = 0
                            self.result = []
                            while teststep * batch_size < tsent2q.shape[0]:
                                ques_in = sent[tsent2q[teststep * batch_size: (teststep + 1) * batch_size]]
                                ans_in = sent[tsent2a[teststep * batch_size: (teststep + 1) * batch_size]]
                                lenques_in = lenwords[tsent2q[teststep * batch_size: (teststep + 1) * batch_size]]
                                lenans_in = lenwords[tsent2a[teststep * batch_size: (teststep + 1) * batch_size]]
                                leng_ques = ques_in.shape[0]
                                if (leng_ques < batch_size):
                                    i = np.array(
                                        [[0 for _i in range(0, senlen)] for _j in range(leng_ques, batch_size)])
                                    j = np.array([1 for _j in range(leng_ques, batch_size)])
                                    ques_in = np.concatenate([ques_in, i], axis=0)
                                    ans_in = np.concatenate([ans_in, i], axis=0)
                                    lenques_in = np.concatenate([lenques_in, j], axis=0)
                                    lenans_in = np.concatenate([lenans_in, j], axis=0)

                                ans_in = np.concatenate((ans_in, ans_in), axis=0)
                                lenans_in = np.concatenate((lenans_in, lenans_in), axis=0)

                                thisr = sess.run(qa1_cos, feed_dict={
                                    self.input_ques: ques_in,
                                    self.input_ans: ans_in,
                                    self.ques_lstm.input_len: lenques_in,
                                    self.ans_lstm.input_len: lenans_in,
                                    self.ques_lstm.keep_prob: 1.0,
                                    self.ans_lstm.keep_prob: 1.0
                                })
                                teststep = teststep + 1
                                self.result += thisr.tolist()
                            outf = open('/home/wanghao/workspace/wikiQA/results/test_%.4f.txt' % cost_all, 'w')
                            for i in range(len(intest2q)):
                                print(intest2q[i], intest2a[i], self.result[i][0], file=outf)
                            outf.close()
                        print('Test finished...')

                    if step % (8800 // batch_size - 2)==0:
                        shuffle_indices = np.random.permutation(np.arange(len(senq_train)))
                        senq_train = senq_train[shuffle_indices]
                        senar_train = senar_train[shuffle_indices]
                        senaw_train = senaw_train[shuffle_indices]
                    if step%10==0:
                        print('Step %3d: %1.2f, %3.4f, %3.4f'%(step, acc, cost, cost_all))
                    step = step + 1
                print('Stop training...', end="")

    def embedding(self):
        self.input_ans = tf.placeholder(tf.int32, shape=(None, None), name='input')
        self.input_ques = tf.placeholder(tf.int32, shape=(None, None), name='input')
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.lstm_in_ans = tf.nn.embedding_lookup(self.wordmat, self.input_ans)
            self.lstm_in_ques = tf.nn.embedding_lookup(self.wordmat, self.input_ques)

    def get_batch(self, sent, lenwords, senq_train, senar_train, senaw_train, step, batch_size):
        count = len(senq_train)
        all_step = count // batch_size - 2
        ques_in = sent[np.array(
            senq_train[(step % all_step) * batch_size:
            (step % all_step + 1) * batch_size])]
        ans_in = sent[np.concatenate((senar_train[(step % all_step) * batch_size:(step % all_step + 1) * batch_size],
                                      senaw_train[(step % all_step) * batch_size:(step % all_step + 1) * batch_size]), axis=0)]
        lenques_in = lenwords[np.array(senq_train[(step % all_step) * batch_size:(step % all_step + 1) * batch_size])]
        lenans_in = lenwords[np.concatenate((senar_train[(step % all_step) * batch_size:(step % all_step + 1) * batch_size],
                                             senaw_train[(step % all_step) * batch_size:(step % all_step + 1) * batch_size]), axis=0)]
        return ques_in, ans_in, lenques_in, lenans_in