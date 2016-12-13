import tensorflow as tf

class SenLSTM:
    def __init__(self, inx, emb_size, senlen, outputlen, wordmat, n_hidden, num_layers, batch_size, inname):
        self.lstm_in = inx
        self.input_len = tf.placeholder(tf.int32, shape=(None), name='input1')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.embedding_size = emb_size
        self.senlen = senlen
        self.hidden = n_hidden

        with tf.variable_scope(inname):
            self.lstm_cell_o = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
            self.lstm_cell = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_o,
                                                           input_keep_prob=self.keep_prob,
                                                           output_keep_prob=self.keep_prob)

            self.lstm_cell_b_o = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
            self.lstm_cell_b = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_b_o,
                                                             input_keep_prob=self.keep_prob,
                                                             output_keep_prob=self.keep_prob)


            lstm_outputs, lstm_states = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell, self.lstm_cell_b,
                                                 self.lstm_in,
                                                 sequence_length=self.input_len,
                                                 dtype=tf.float32)
            lstm_sf, lstm_sb = lstm_states
            lstm_sf = lstm_sf[1]
            lstm_sb = lstm_sb[1]
            lstm_sout = tf.concat(1, [lstm_sf, lstm_sb])
            self.lstm_out = lstm_sout

    # unused function to get the last relevant output of LSTM network
    def last_relevant(self, output, length):
        batch_size = tf.shape(output[0])[0]
        max_length = self.senlen
        out_size = tf.shape(output[0])[1]
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant