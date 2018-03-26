import tensorflow as tf


class AttentionModel(object):
    def __init__(self,
                 n_disease,
                 n_drug,
                 n_visit,
                 n_disease_category,
                 n_drug_category,
                 n_embed=500,
                 n_rnn=(200, 200),
                 dropout_rate=0.2,
                 batch_size=100):
        self.n_disease = n_disease
        self.n_drug = n_drug
        self.n_visit = n_visit
        self.n_disease_category = n_disease_category
        self.n_drug_category = n_drug_category
        self.n_embed = n_embed
        self.n_rnn = n_rnn
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

    def gru_unit(self, hidden_size, dropout_rate):
        gru_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
        gru_cell = tf.nn.rnn_cell.DropoutWrapper(cell=gru_cell, input_keep_prob=1.0, output_keep_prob=1 - dropout_rate)
        return gru_cell

    def gru_unit_0(self, hidden_size, dropout_rate):
        gru_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size, name="0")
        gru_cell = tf.nn.rnn_cell.DropoutWrapper(cell=gru_cell, input_keep_prob=1.0, output_keep_prob=1 - dropout_rate)
        return gru_cell
    # def build_model(self, x_disease, x_drug, disease_label, drug_label):
    #     x_code = tf.concat([x_disease, x_drug], axis=2)
    #     x_code = tf.reshape(x_code, [-1, self.n_disease + self.n_drug])
    #     w_embed = tf.get_variable('w_embed',
    #                               shape=[self.n_disease + self.n_drug, self.n_embed])
    #     b_embed = tf.get_variable('b_mebed', shape=[self.n_embed])
    #     hidden = tf.nn.relu(tf.add(tf.matmul(x_code, w_embed), b_embed))
    #     hidden = tf.reshape(hidden, [-1, self.n_visit, self.n_embed])
    #     stacked_gru = tf.nn.rnn_cell.MultiRNNCell(
    #         [self.gru_unit(self.n_rnn[0], self.dropout_rate) for i in range(len(self.n_rnn))], state_is_tuple=True)
    #     outputs, states = tf.nn.dynamic_rnn(cell=stacked_gru, inputs=hidden, dtype=tf.float32)
    #     w_alpha = tf.get_variable('w_alpha',
    #                               shape=[self.n_rnn[-1], 1])
    #     b_alpha = tf.get_variable('b_alpha', shape=[1])
    #     w_code = tf.get_variable('w_code',
    #                              shape=[self.n_rnn[-1], self.n_disease_category + self.n_drug_category])
    #     b_code = tf.get_variable('b_code', shape=[self.n_disease_category + self.n_drug_category])
    #     code_loss = tf.Tensor
    #     code_y_last = tf.Tensor
    #     code_label = tf.concat([disease_label, drug_label], axis=2)
    #     for i in range(1, self.n_visit):
    #         outputs_reshape = tf.reshape(outputs, [-1, self.n_rnn[-1]])
    #         alpha = tf.reshape(
    #             tf.reduce_mean(tf.nn.softmax(tf.reshape(tf.add(tf.matmul(outputs_reshape, w_alpha), b_alpha),
    #                                                     [-1, self.n_visit])[:, 0:i]), axis=1), [-1, 1])
    #         h_ = alpha * outputs[:, i, :]
    #         code_loss = tf.reduce_sum(
    #             tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.add(tf.matmul(h_, w_code), b_code),
    #                                                        labels=code_label[:, i, :]))
    #         code_y_last = tf.nn.softmax(tf.add(tf.matmul(h_, w_code), b_code))


    def build_model(self, x_disease, x_drug, disease_label, drug_label):
        x_code = tf.concat([x_disease, x_drug], axis=2)
        x_code = tf.reshape(x_code, [-1, self.n_disease + self.n_drug])
        w_embed = tf.get_variable('w_embed',
                                  shape=[self.n_disease + self.n_drug, self.n_embed])
        b_embed = tf.get_variable('b_mebed', shape=[self.n_embed])
        hidden = tf.nn.relu(tf.add(tf.matmul(x_code, w_embed), b_embed))
        hidden = tf.reshape(hidden, [-1, self.n_visit, self.n_embed])
        stacked_gru = tf.nn.rnn_cell.MultiRNNCell(
            [self.gru_unit(self.n_rnn[0], self.dropout_rate) for i in range(len(self.n_rnn))], state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(cell=stacked_gru, inputs=hidden, dtype=tf.float32)
        stacked_gru_0 = tf.nn.rnn_cell.MultiRNNCell(
            [self.gru_unit_0(self.n_rnn[0], self.dropout_rate) for i in range(len(self.n_rnn))], state_is_tuple=True)
        outputs_0, states_0 = tf.nn.dynamic_rnn(cell=stacked_gru_0, inputs=hidden, dtype=tf.float32)
        w_alpha = tf.get_variable('w_alpha',
                                  shape=[self.n_rnn[-1], 1])
        b_alpha = tf.get_variable('b_alpha', shape=[1])
        w_code = tf.get_variable('w_code',
                                 shape=[self.n_rnn[-1], self.n_disease_category + self.n_drug_category])
        b_code = tf.get_variable('b_code', shape=[self.n_disease_category + self.n_drug_category])
        code_loss = tf.Tensor
        code_y_last = tf.Tensor
        code_label = tf.concat([disease_label, drug_label], axis=2)
        for i in range(1, self.n_visit):
            outputs_reshape = tf.reshape(outputs, [-1, self.n_rnn[-1]])
            alpha = tf.reshape(
                tf.reduce_mean(tf.nn.softmax(tf.reshape(tf.add(tf.matmul(outputs_reshape, w_alpha), b_alpha),
                                                        [-1, self.n_visit])[:, 0:i]), axis=1), [-1, 1])
            h_ = alpha * outputs_0[:, i, :]
            code_loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.add(tf.matmul(h_, w_code), b_code),
                                                           labels=code_label[:, i, :]))
            code_y_last = tf.nn.softmax(tf.add(tf.matmul(h_, w_code), b_code))


        # x_code = tf.concat([x_disease, x_drug], axis=2)
        # x_code = tf.reshape(x_code, [-1, self.n_disease + self.n_drug])
        # w_embed = tf.get_variable('w_embed',
        #                           shape=[self.n_disease + self.n_drug, self.n_embed])
        # b_embed = tf.get_variable('b_mebed', shape=[self.n_embed])
        # hidden = tf.nn.relu(tf.add(tf.matmul(x_code, w_embed), b_embed))
        # hidden = tf.reshape(hidden, [-1, self.n_visit, self.n_embed])
        # stacked_gru = tf.nn.rnn_cell.MultiRNNCell(
        #     [self.gru_unit(self.n_rnn[0], self.dropout_rate) for i in range(len(self.n_rnn))], state_is_tuple=True)
        # outputs, states = tf.nn.dynamic_rnn(cell=stacked_gru, inputs=hidden, dtype=tf.float32)
        # w_alpha = tf.get_variable('w_alpha',
        #                           shape=[self.n_rnn[-1], 1])
        # b_alpha = tf.get_variable('b_alpha', shape=[1])
        # w_code = tf.get_variable('w_code',
        #                          shape=[self.n_rnn[-1], self.n_disease_category + self.n_drug_category])
        # b_code = tf.get_variable('b_code', shape=[self.n_disease_category + self.n_drug_category])
        # code_loss = tf.Tensor
        # code_y_last = tf.Tensor
        # code_label = tf.concat([disease_label, drug_label], axis=2)
        # for i in range(1, self.n_visit):
        #     outputs_reshape = tf.reshape(outputs, [-1, self.n_rnn[-1]])
        #     alpha = tf.reshape(tf.nn.softmax(tf.reshape(tf.add(tf.matmul(outputs_reshape, w_alpha), b_alpha),
        #                                                 [-1, self.n_visit])[:, 0:i]), [-1, i, 1])
        #     c = tf.reduce_sum(alpha * outputs[:, 0:i, :], axis=1)
        #     h_ = tf.concat([c, outputs[:, i, :]], axis=1)
        #     # h_ = c * outputs[:, i, :]
        #     code_loss = tf.reduce_sum(
        #         tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.add(tf.matmul(h_, w_code), b_code),
        #                                                    labels=code_label[:, i, :]))
        #     code_y_last = tf.nn.softmax(tf.add(tf.matmul(h_, w_code), b_code))


        return code_loss, code_y_last

    def print2file(self, content, out_file):
        outfd = open(out_file, 'a')
        outfd.write(content + '\n')
        outfd.close()
