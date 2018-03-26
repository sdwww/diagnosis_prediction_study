import time

import numpy as np
import tensorflow as tf

from doctor_ai.Model import Model
from doctor_ai.Config import get_config
from doctor_ai.Standard import precision_top


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
               dataset_disease_path='',
               label_disease_path='',
               dataset_drug_path='',
               label_drug_path='',
               model_path='',
               out_path='',
               save_max_keep=0):
    x_disease = tf.placeholder(tf.float32, [None, model.n_visit, model.n_disease])
    x_drug = tf.placeholder(tf.float32, [None, model.n_visit, model.n_drug])
    y_disease = tf.placeholder(tf.float32, [None, model.n_visit, model.n_disease_category])
    y_drug = tf.placeholder(tf.float32, [None, model.n_visit, model.n_drug_category])
    code_loss, code_y = model.build_model(x_disease, x_drug, y_disease, y_drug)
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.005)
    # optimize = optimizer.minimize(loss=code_loss)
    #
    # init_op = tf.global_variables_initializer()
    dataset_disease_test, dataset_drug_test, label_disease_test, label_drug_test \
        = load_test_data(dataset_disease_path, label_disease_path, dataset_drug_path, label_drug_path)
    saver = tf.train.Saver(max_to_keep=save_max_keep)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        n_test_batches = int(np.ceil(float(dataset_disease_test.shape[0])) / float(model.batch_size))
        idx = np.arange(dataset_disease_test.shape[0])

        # 测试集进行验证
        idx = np.arange(len(dataset_disease_test))
        top_precision_vec = []
        for i in range(n_test_batches):
            batch_idx = np.random.choice(idx, size=model.batch_size, replace=False)
            batch_x_disease = dataset_disease_test[batch_idx]
            batch_x_drug = dataset_drug_test[batch_idx]
            batch_y_disease = label_disease_test[batch_idx]
            batch_y_drug = label_drug_test[batch_idx]
            predict_code, = sess.run([code_y],
                                     feed_dict={x_disease: batch_x_disease, x_drug: batch_x_drug,
                                                y_disease: batch_y_disease, y_drug: batch_y_drug})
            top_precision_vec.append(
                precision_top(np.concatenate([batch_y_disease[:, -1, :], batch_y_drug[:, -1, :]], axis=1),
                              predict_code))
        print("test_top_precision:", np.mean(top_precision_vec, axis=0))
        print("sum_top_precision:", np.sum(np.mean(top_precision_vec, axis=0)))


if __name__ == '__main__':
    start = time.clock()
    config = get_config()
    med_model = Model(n_disease=config['n_disease'],
                               n_drug=config['n_drug'],
                               n_visit=config['n_visit'],
                               n_disease_category=config['n_disease_category'],
                               n_drug_category=config['n_drug_category'],
                               n_embed=config['n_embed'],
                               n_rnn=config['n_rnn'],
                               dropout_rate=config['dropout_rate'],
                               batch_size=config['batch_size'])
    test_model(med_model,
               dataset_disease_path=config['disease_file'],
               label_disease_path=config['disease_label'],
               dataset_drug_path=config['drug_file'],
               label_drug_path=config['drug_label'],
               model_path=config['model_file'],
               out_path=config['out_file'],
               save_max_keep=config['save_max_keep'])
    print(time.clock() - start)
# 0.53514286 0.63885714 0.69771429 0.742      0.78114286
# 0.81314286 0.83885714 0.854      0.86914286 0.88285714