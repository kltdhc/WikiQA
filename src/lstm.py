import tensorflow as tf
import numpy as np

class SenLSTM:
    def __init__(self, emb_size, senlen, outputlen, wordmat, n_hidden, num_layers, batch_size, inname):
        self.input_x = tf.placeholder(tf.int32, shape=(None, None), name='input')
        self.input_len = tf.placeholder(tf.int32, shape=(None), name='input1')
        self.embedding_size = emb_size
        self.senlen = senlen
        self.hidden = n_hidden
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.wordmat = wordmat
            self.lstm_in1 = tf.nn.embedding_lookup(self.wordmat, self.input_x)
            # Permuting batch_size and n_steps
            self.lstm_in2 = tf.transpose(self.lstm_in1, [1, 0, 2])
            # Reshaping to (n_steps*batch_size, n_input)
            self.lstm_in3 = tf.reshape(self.lstm_in2, [-1, self.embedding_size])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            self.lstm_in= tf.split(0, senlen, self.lstm_in3)

        with tf.variable_scope(inname):
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=0.2, state_is_tuple=True)
            self.lstm_cell = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell, input_keep_prob=0.7,
                                                           output_keep_prob=0.7)

            self.lstm_cell_b = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=0.2, state_is_tuple=True)
            self.lstm_cell_b = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_b, input_keep_prob=0.7,
                                                           output_keep_prob=0.7)

            # self.outweight = tf.Variable(tf.random_normal([n_hidden, outputlen]), name='outw')
            self.outweight = tf.Variable(tf.random_normal([n_hidden*2, outputlen*4]), name='outw')
            self.outbias = tf.Variable(tf.random_normal([outputlen*4]))
            # self.lstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell] * num_layers, state_is_tuple=True)
            # _initial_state = self.lstm_cell.zero_state(batch_size, tf.float32)
            # lstm_outputs, self.state = tf.nn.rnn(self.lstm_cell,
            lstm_outputs, _, _ = tf.nn.bidirectional_rnn(self.lstm_cell, self.lstm_cell_b,
                                                 self.lstm_in,
                                                 sequence_length=self.input_len,
                                                 dtype=tf.float32)
            # self.lstm_out = lstm_outputs[self.input_len]
            # _, lstm_out = self.state
            lstm_out = self.last_relevant(lstm_outputs, self.input_len)
            # lstm_out = lstm_outputs[-1]
            lstm_out = tf.matmul(lstm_out, self.outweight)+self.outbias
            lstm_out = tf.reshape(lstm_out, [-1, 10, 10, 1])
            lstm_out = tf.nn.max_pool(lstm_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            self.lstm_out = tf.reshape(lstm_out, [-1, 25])

    def last_relevant(self, output, length):
        batch_size = tf.shape(output[0])[0]
        max_length = self.senlen
        out_size = tf.shape(output[0])[1]
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant