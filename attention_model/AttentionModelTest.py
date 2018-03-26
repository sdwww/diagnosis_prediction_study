import time

import numpy as np
import tensorflow as tf

from attention_model.AttentionModel import AttentionModel
from attention_model.Config import get_config
from attention_model.Standard import precision_top


def load_test_data(dataset_disease_path='',
                   label_disease_path='',
                   dataset_drug_path='',
                   label_drug_path=''):
    dataset_disease = np.load(dataset_disease_path)
    dataset_disease_test = dataset_disease['test']
    label_disease = np.load(label_disease_path)
    label_disease_test = label_disease['test']
    dataset_drug = np.load(dataset_drug_path)
    dataset_drug_test = dataset_drug['test']
    label_drug = np.load(label_drug_path)
    label_drug_test = label_drug['test']
    return dataset_disease_test, dataset_drug_test, label_disease_test, label_drug_test


def test_model(model,
               n_disease_category,
               n_drug_category,
               dataset_disease_path='',
               label_disease_path='',
               dataset_drug_path='',
               label_drug_path='',
               model_path='', ):
    x_disease = tf.placeholder(tf.float32, [None, model.n_visit, model.n_disease])
    x_drug = tf.placeholder(tf.float32, [None, model.n_visit, model.n_drug])
    y_disease = tf.placeholder(tf.float32, [None, model.n_visit, model.n_disease_category])
    y_drug = tf.placeholder(tf.float32, [None, model.n_visit, model.n_drug_category])
    code_loss, code_y = model.build_model(x_disease, x_drug, y_disease, y_drug)

    dataset_disease_test, dataset_drug_test, label_disease_test, label_drug_test \
        = load_test_data(dataset_disease_path, label_disease_path, dataset_drug_path, label_drug_path)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 测试集
        predict_code, = sess.run([code_y],
                                 feed_dict={x_disease: dataset_disease_test, x_drug: dataset_drug_test,
                                            y_disease: label_disease_test, y_drug: label_drug_test})
        top_precision_vec = precision_top(
            np.concatenate([label_disease_test[:, -1, :], label_drug_test[:, -1, :]], axis=1),
            predict_code)
        top_precision_disease_vec = precision_top(label_disease_test[:, -1, :],
                                                  predict_code[:, 0:n_disease_category])
        top_precision_drug_vec = precision_top(label_drug_test[:, -1, :],
                                               predict_code[:, n_disease_category:])
        print("test_top_precision_disease:", top_precision_disease_vec)
        print("test_top_precision_drug:", top_precision_drug_vec)
        print("test_top_precision:", top_precision_vec)
        print("sum_top_precision:", np.sum(np.mean(top_precision_vec, axis=0)))


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
    test_model(med_model,
               config['n_disease_category'],
               config['n_drug_category'],
               dataset_disease_path=config['disease_file'],
               label_disease_path=config['disease_label'],
               dataset_drug_path=config['drug_file'],
               label_drug_path=config['drug_label'],
               model_path=config['model_file'],
               )
    print(time.clock() - start)

# test_top_precision: [0.594      0.62314286 0.652      0.67942857 0.71714286 0.754
#  0.78428571 0.81657143 0.84342857 0.87028571]   2rnn
