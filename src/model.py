import lstm
import tensorflow as tf
import numpy as np
import sys

class train_model:
    def __init__(self, senlen, wordmat, sen_q, sen_ar, sen_aw, sentences, senlens, testq, testa):
        outputlen = 25
        n_hidden = 100
        batch_size = 100
        learning_rate = 2e-1
        training_iter = 60000
        embedding_size = len(wordmat[0])
        # shuffle data
        senq = np.array(sen_q)
        senar = np.array(sen_ar)
        senaw = np.array(sen_aw)
        sent = np.array(sentences)
        tsentq = np.array(testq)
        tsenta = np.array(testa)
        lenwords = np.array(senlens)
        np.random.seed(100)
        shuffle_indices = np.random.permutation(np.arange(len(senq)))
        senq_shuffled = senq[shuffle_indices]
        senar_shuffled = senar[shuffle_indices]
        senaw_shuffled = senaw[shuffle_indices]
        senq_train, senq_test = senq_shuffled, senq_shuffled[-900:]
        senar_train, senar_test = senar_shuffled, senar_shuffled[-900:]
        senaw_train, senaw_test = senaw_shuffled, senaw_shuffled[-900:]
        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                # basic cells
                self.wordmat = tf.Variable(wordmat, name='wordmat')
                self.ans_lstm = lstm.SenLSTM(embedding_size, senlen,
                                             outputlen, self.wordmat,
                                             n_hidden, 2, batch_size*2, 'anslstm')
                self.ques_lstm = lstm.SenLSTM(embedding_size, senlen,
                                              outputlen, self.wordmat,
                                              n_hidden, 2, batch_size, 'queslstm')
                step = 0
                # cosine similarity of the vectors
                ans1 = tf.slice(self.ans_lstm.lstm_out, [0, 0], [batch_size, outputlen])
                ans2 = tf.slice(self.ans_lstm.lstm_out, [batch_size, 0], [batch_size, outputlen])
                ques = self.ques_lstm.lstm_out
                qa1 = tf.matmul(tf.mul(ques, ans1), tf.ones([outputlen, 1], tf.float32))
                qa2 = tf.matmul(tf.mul(ques, ans2), tf.ones([outputlen, 1], tf.float32))
                a1sq = tf.matmul(tf.sqrt(tf.mul(ans1, ans1)), tf.ones([outputlen, 1], tf.float32))
                a2sq = tf.matmul(tf.sqrt(tf.mul(ans2, ans2)), tf.ones([outputlen, 1], tf.float32))
                qsq = tf.matmul(tf.sqrt(tf.mul(ques, ques)), tf.ones([outputlen, 1], tf.float32))
                qa1_cos = tf.div(qa1, tf.mul(a1sq, qsq))
                qa2_cos = tf.div(qa2, tf.mul(a2sq, qsq))
                loss = tf.matmul(tf.ones([1, batch_size], tf.float32),
                                 tf.maximum(tf.zeros([batch_size, 1], tf.float32),
                                            1.9 * tf.ones([batch_size, 1], tf.float32) + qa2_cos - qa1_cos))
                loss_all =  1e-3 * tf.nn.l2_loss(self.ans_lstm.outweight)
                loss_all += 1e-3 * tf.nn.l2_loss(self.ans_lstm.outbias)
                loss_all += 1e-3 * tf.nn.l2_loss(self.ques_lstm.outweight)
                loss_all += 1e-3 * tf.nn.l2_loss(self.ques_lstm.outbias)
                loss_all = loss + loss_all
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss_all)
                pred = tf.argmax(tf.concat(1, [qa1_cos, qa2_cos]),1)
                accuracy = tf.reduce_mean(tf.cast(
                        tf.equal(pred, tf.zeros([batch_size, 1], tf.int64)),
                        tf.float32))
                step = 1
                init = tf.initialize_all_variables()
                sess.run(init)
                print('Model Defination: Done!')
                while step*batch_size<training_iter:
                    ques_in = sent[np.array(senq_train[(step % 88) * 100:(step % 88 + 1) * 100])]
                    ans_in = sent[np.concatenate((senar_train[(step % 88) * 100:(step % 88 + 1) * 100],
                                      senaw_train[(step % 88) * 100:(step % 88 + 1) * 100]), axis=0 )]
                    lenques_in = lenwords[np.array(senq_train[(step % 88) * 100:(step % 88 + 1) * 100])]
                    lenans_in = lenwords[np.concatenate((senar_train[(step % 88) * 100:(step % 88 + 1) * 100],
                                      senaw_train[(step % 88) * 100:(step % 88 + 1) * 100]), axis=0 )]
                    # print('lenquesin', lenques_in)
                    # print('lenansin', lenans_in)
                    # print(ques_in)
                    # print(sess.run(self.wordmat))
                    # print('lstm_out', sess.run(self.ques_lstm.lstm_out, feed_dict={
                    #    self.ques_lstm.input_x: ques_in,
                    #    self.ans_lstm.input_x: ans_in,
                    #    self.ques_lstm.input_len: lenques_in,
                    #    self.ans_lstm.input_len: lenans_in
                    # }))
                    # break

                    _, cost = sess.run([optimizer, loss], feed_dict={
                                       self.ques_lstm.input_x:ques_in,
                                       self.ans_lstm.input_x:ans_in,
                                       self.ques_lstm.input_len: lenques_in,
                                       self.ans_lstm.input_len: lenans_in
                                   })
                    if step%80==0:
                        shuffle_indices = np.random.permutation(np.arange(len(senq_train)))
                        senq_train = senq_train[shuffle_indices]
                        senar_train = senar_train[shuffle_indices]
                        senaw_train = senaw_train[shuffle_indices]
                    if step%10==0:
                        acc = sess.run(accuracy, feed_dict={
                            self.ques_lstm.input_x: ques_in,
                            self.ans_lstm.input_x: ans_in,
                            self.ques_lstm.input_len: lenques_in,
                            self.ans_lstm.input_len: lenans_in
                        })
                        cost, cost_all = sess.run([loss, loss_all], feed_dict={
                            self.ques_lstm.input_x: ques_in,
                            self.ans_lstm.input_x: ans_in,
                            self.ques_lstm.input_len: lenques_in,
                            self.ans_lstm.input_len: lenans_in
                        })
                        #test = sess.run(self.ans_lstm.lstm_in, feed_dict={
                        #    self.ques_lstm.input_x: ques_in,
                        #    self.ans_lstm.input_x: ans_in
                        #})
                        #print(test)
                        print('Step %3d: %1.2f, %3.4f, %3.4f'%(step, acc, cost, cost_all))
                    step = step + 1
                print('stop training')
                # step = 0
                # fout = open("/home/wanghao/workspace/wikiQA/eval_shuffle", 'w')
                # while step*batch_size<900:
                #     ques_in = sent[np.array(senq_test[(step) * batch_size:(step + 1) * batch_size])]
                #     ans_in = sent[np.concatenate((senar_test[(step) * batch_size:(step + 1) * batch_size],
                #                                   senaw_test[(step) * batch_size:(step + 1) * batch_size]), axis=0)]
                #     lenques_in = lenwords[np.array(senq_test[(step) * batch_size:(step + 1) * batch_size])]
                #     lenans_in = lenwords[np.concatenate((senar_test[(step) * batch_size:(step + 1) * batch_size],
                #                                          senaw_test[(step) * batch_size:(step + 1) * batch_size]), axis=0)]
                #     print('step', step, file=fout)
                #     print('eval step', step)
                #     print(sess.run([qa1_cos, qa2_cos], feed_dict={
                #         self.ques_lstm.input_x: ques_in,
                #         self.ans_lstm.input_x: ans_in,
                #         self.ques_lstm.input_len: lenques_in,
                #         self.ans_lstm.input_len: lenans_in
                #     }), file=fout)
                #     step += 1
                # fout.close()
                step = 0
                self.result = []
                while step*batch_size<tsentq.shape[0]:
                    ques_in = sent[tsentq[step*batch_size: (step+1)*batch_size]]
                    ans_in = sent[tsenta[step*batch_size: (step+1)*batch_size]]
                    lenques_in = lenwords[tsentq[step * batch_size: (step + 1) * batch_size]]
                    lenans_in = lenwords[tsenta[step * batch_size: (step + 1) * batch_size]]
                    leng_ques = ques_in.shape[0]
                    if(leng_ques<100):
                        i = np.array([[0 for _i in range(0, senlen)] for _j in range(leng_ques, batch_size)])
                        j = np.array([1 for _j in range(leng_ques, batch_size)])
                        ques_in = np.concatenate([ques_in, i], axis=0)
                        ans_in = np.concatenate([ans_in, i], axis=0)
                        lenques_in = np.concatenate([lenques_in, j], axis=0)
                        lenans_in = np.concatenate([lenans_in, j], axis=0)
                    thisr = sess.run(qa1_cos, feed_dict={
                        self.ques_lstm.input_x: ques_in,
                        self.ans_lstm.input_x: ans_in,
                        self.ques_lstm.input_len: lenques_in,
                        self.ans_lstm.input_len: lenans_in
                    })
                    step = step+1
                    self.result += thisr.tolist()