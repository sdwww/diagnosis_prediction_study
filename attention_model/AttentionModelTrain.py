import time

import numpy as np
import tensorflow as tf

from attention_model.AttentionModel import AttentionModel
from attention_model.Config import get_config
from attention_model.Standard import precision_top


def load_train_data(dataset_disease_path='',
                    label_disease_path='',
                    dataset_drug_path='',
                    label_drug_path=''):
    dataset_disease = np.load(dataset_disease_path)
    dataset_disease_train = dataset_disease['train']
    dataset_disease_valid = dataset_disease['valid']
    label_disease = np.load(label_disease_path)
    label_disease_train = label_disease['train']
    label_disease_valid = label_disease['valid']
    dataset_drug = np.load(dataset_drug_path)
    dataset_drug_train = dataset_drug['train']
    dataset_drug_valid = dataset_drug['valid']
    label_drug = np.load(label_drug_path)
    label_drug_train = label_drug['train']
    label_drug_valid = label_drug['valid']
    return dataset_disease_train, dataset_disease_valid, dataset_drug_train, dataset_drug_valid, \
           label_disease_train, label_disease_valid, label_drug_train, label_drug_valid


def train_model(model,
                dataset_disease_path='',
                label_disease_path='',
                dataset_drug_path='',
                label_drug_path='',
                out_path='',
                n_epochs=20,
                save_max_keep=0):
    x_disease = tf.placeholder(tf.float32, [None, model.n_visit, model.n_disease])
    x_drug = tf.placeholder(tf.float32, [None, model.n_visit, model.n_drug])
    y_disease = tf.placeholder(tf.float32, [None, model.n_visit, model.n_disease_category])
    y_drug = tf.placeholder(tf.float32, [None, model.n_visit, model.n_drug_category])
    code_loss, code_y = model.build_model(x_disease, x_drug, y_disease, y_drug)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
    optimize = optimizer.minimize(loss=code_loss)

    init_op = tf.global_variables_initializer()
    dataset_disease_train, dataset_disease_valid, dataset_drug_train, dataset_drug_valid, \
    label_disease_train, label_disease_valid, label_drug_train, label_drug_valid \
        = load_train_data(dataset_disease_path, label_disease_path, dataset_drug_path, label_drug_path)
    saver = tf.train.Saver(max_to_keep=save_max_keep)
    log_file = out_path + '.log'

    with tf.Session() as sess:
        # create a log writer. run 'tensorboard --logdir=./logs'
        # writer = tf.summary.FileWriter("./logs", sess.graph)  # for 1.0
        sess.run(init_op)
        n_train_batches = int(np.ceil(float(dataset_disease_train.shape[0])) / float(model.batch_size))
        n_valid_batches = int(np.ceil(float(dataset_disease_valid.shape[0])) / float(model.batch_size))

        # 训练集进行训练
        idx = np.arange(dataset_disease_train.shape[0])
        last_epoch = 0
        for epoch in range(n_epochs):
            last_epoch = epoch
            code_loss_vec = []
            print("epoch:", epoch + 1)
            for i in range(n_train_batches):
                batch_idx = np.random.choice(idx, size=model.batch_size, replace=False)
                batch_x_disease = dataset_disease_train[batch_idx]
                batch_x_drug = dataset_drug_train[batch_idx]
                batch_y_disease = label_disease_train[batch_idx]
                batch_y_drug = label_drug_train[batch_idx]
                _, loss_code, = sess.run(
                    [optimize, code_loss],
                    feed_dict={x_disease: batch_x_disease, x_drug: batch_x_drug,
                               y_disease: batch_y_disease, y_drug: batch_y_drug, })
                code_loss_vec.append(loss_code)
            print("code_loss:", np.mean(code_loss_vec))

            # 验证集进行验证
            idx = np.arange(len(dataset_disease_valid))
            top_precision_vec = []
            for i in range(n_valid_batches):
                batch_idx = np.random.choice(idx, size=model.batch_size, replace=False)
                batch_x_disease = dataset_disease_valid[batch_idx]
                batch_x_drug = dataset_drug_valid[batch_idx]
                batch_y_disease = label_disease_valid[batch_idx]
                batch_y_drug = label_drug_valid[batch_idx]
                predict_code, = sess.run(
                    [code_y],
                    feed_dict={x_disease: batch_x_disease, x_drug: batch_x_drug,
                               y_disease: batch_y_disease, y_drug: batch_y_drug})
                top_precision_vec.append(
                    precision_top(np.concatenate([batch_y_disease[:, -1, :], batch_y_drug[:, -1, :]], axis=1),
                                  predict_code))
            print("valid_top_precision:", np.mean(top_precision_vec, axis=0))
            # self.print2file(buffer, log_file)
        save_path = saver.save(sess, out_path, global_step=last_epoch)
        print(save_path)


if __name__ == '__main__':
    start = time.clock()
    config = get_config()
    med_model = AttentionModel(n_disease=config['n_disease'],
                               n_drug=config['n_drug'],
                               n_visit=config['n_visit'],
                               n_disease_category=config['n_disease_category'],
                               n_drug_category=config['n_drug_category'],
                               n_embed=config['n_embed'],
                               n_rnn=config['n_rnn'],
                               dropout_rate=config['dropout_rate'],
                               batch_size=config['batch_size'])
    train_model(med_model,
                dataset_disease_path=config['disease_file'],
                label_disease_path=config['disease_label'],
                dataset_drug_path=config['drug_file'],
                label_drug_path=config['drug_label'],
                out_path=config['out_file'],
                n_epochs=config['n_epoch'],
                save_max_keep=config['save_max_keep'])
    print(time.clock() - start)
