import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

from data_process import DBOptions

patient_num = 35308
visit_num = 50
disease_num = 1332
drug_num = 1125
disease_category_num = 612
drug_category_num = 136
valid_rate = 0.11111
test_rate = 0.1


def dataset_split(dataset):
    train_data, test_data = train_test_split(dataset, test_size=test_rate, random_state=0)
    train_data, valid_data = train_test_split(train_data, test_size=valid_rate, random_state=0)
    return train_data, valid_data, test_data


def create_info(db_cursor):
    # 创建个人信息的数据输入
    sql = 'select GRBH_INDEX,GENDER,AGE from DATA_INFO'
    all_info = DBOptions.get_sql(sql, cursor=db_cursor)
    dataset_info = np.zeros((patient_num, 4), dtype='int16')
    print('信息数据集大小：', dataset_info.shape)
    for i in all_info:
        if i[1]:
            dataset_info[int(i[0]), 0] = int(i[1])
        if not i[1]:
            dataset_info[int(i[0]), 1] = 1
        if i[2]:
            dataset_info[int(i[0]), 2] = int(i[2])
        if not i[2]:
            dataset_info[int(i[0]), 3] = 1
    train_data, valid_data, test_data = dataset_split(dataset_info)
    np.savez_compressed('../dataset/dataset_info.npz', train=train_data, valid=valid_data, test=test_data)


def create_disease(db_cursor):
    # 创建诊断数据的输入
    sql = 'select grbh_index,xh_index,jbbm_index from DATA_JBBM where XH_INDEX!=0'
    all_info = DBOptions.get_sql(sql, cursor=db_cursor)
    dataset_disease = np.zeros((patient_num, visit_num, disease_num), dtype='int16')
    print('诊断数据输入大小：', dataset_disease.shape)
    for i in all_info:
        if i[1] <= visit_num:
            dataset_disease[i[0], -i[1], i[2]] = 1
    train_data, valid_data, test_data = dataset_split(dataset_disease)
    np.savez_compressed('../dataset/dataset_disease.npz', train=train_data, valid=valid_data, test=test_data)

    # 创建诊断数据的标签
    sql = "select grbh_index,xh_index,jbbm_index from DATA_JBBM"
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    label_disease = np.zeros((patient_num, visit_num, disease_num), dtype='int16')
    print('诊断数据标签大小：', label_disease.shape)
    for i in all_info:
        if i[1] < visit_num:
            label_disease[i[0], -i[1] - 1, i[2]] = 1
    train_data, valid_data, test_data = dataset_split(label_disease)
    np.savez_compressed('../dataset/label_disease.npz', train=train_data, valid=valid_data, test=test_data)

    # 创建诊断类别数据的标签
    sql = "select grbh_index,xh_index,jbmc_categ_index from DATA_JBBM"
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    label_disease_categ = np.zeros((patient_num, visit_num, disease_category_num), dtype='int16')
    print('诊断类别数据标签大小：', label_disease_categ.shape)
    for i in all_info:
        if i[1] < visit_num:
            label_disease_categ[i[0], -i[1] - 1, i[2]] = 1
    train_data, valid_data, test_data = dataset_split(label_disease_categ)
    np.savez_compressed('../dataset/label_disease_categ.npz', train=train_data, valid=valid_data, test=test_data)


def create_drug(db_cursor):
    # 创建药品数据的输入
    sql = 'select grbh_index,xh_index,DRUG_INDEX from DATA_DRUG,DATA_JBBM where ' \
          'DATA_JBBM.XH =DATA_DRUG.XH AND DATA_JBBM.XH_INDEX!=0'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    dataset_drug = np.zeros((patient_num, visit_num, drug_num), dtype='int16')
    print('药品数据输入大小：', dataset_drug.shape)
    for i in all_info:
        if i[1] <= visit_num:
            dataset_drug[i[0], -i[1], i[2]] = 1
    train_data, valid_data, test_data = dataset_split(dataset_drug)
    np.savez_compressed('../dataset/dataset_drug.npz', train=train_data, valid=valid_data, test=test_data)

    # 创建药品数据的标签
    sql = 'select grbh_index,xh_index,DRUG_INDEX from DATA_DRUG,DATA_JBBM where ' \
          'DATA_JBBM.XH =DATA_DRUG.XH'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    label_drug = np.zeros((patient_num, visit_num, drug_num), dtype='int16')
    print('药品数据标签大小：', label_drug.shape)
    for i in all_info:
        if i[1] < visit_num:
            label_drug[i[0], -i[1] - 1, i[2]] = 1
    train_data, valid_data, test_data = dataset_split(label_drug)
    np.savez_compressed('../dataset/label_drug.npz', train=train_data, valid=valid_data, test=test_data)

    # 创建药品类别数据的标签
    sql = 'select grbh_index,xh_index,DRUG_CATEG_INDEX from DATA_DRUG,DATA_JBBM where ' \
          'DATA_JBBM.XH =DATA_DRUG.XH'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    label_drug_categ = np.zeros((patient_num, visit_num, drug_category_num), dtype='int16')
    print('药品类别数据标签大小：', label_drug_categ.shape)
    for i in all_info:
        if i[1] < visit_num:
            label_drug_categ[i[0], -i[1] - 1, i[2]] = 1
    train_data, valid_data, test_data = dataset_split(label_drug_categ)
    np.savez_compressed('../dataset/label_drug_categ.npz', train=train_data, valid=valid_data, test=test_data)


def create_duration(db_cursor):
    # 创建时间间隔数据的输入
    sql = 'select GRBH_INDEX,XH_INDEX,DURATION from DATA_JBBM where XH_INDEX!=0 ' \
          'group by GRBH_INDEX,XH_INDEX,DURATION order by GRBH_INDEX,XH_INDEX'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    dataset_duration = np.zeros((patient_num, visit_num,1), dtype='int16')
    print('时间间隔数据输入大小：', dataset_duration.shape)
    for i in all_info:
        if i[1] <= visit_num:
            dataset_duration[i[0], -i[1]] = i[2]
    train_data, valid_data, test_data = dataset_split(dataset_duration)
    np.savez_compressed('../dataset/dataset_duration.npz', train=train_data, valid=valid_data, test=test_data)

    # 创建时间间隔数据的标签
    sql = 'select GRBH_INDEX,XH_INDEX,DURATION from DATA_JBBM ' \
          'group by GRBH_INDEX,XH_INDEX,DURATION order by GRBH_INDEX,XH_INDEX'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    label_duration = np.zeros((patient_num, visit_num,1), dtype='int16')
    print('时间间隔数据标签大小：', label_duration.shape)
    for i in all_info:
        if i[1] < visit_num:
            label_duration[i[0], -i[1] - 1] = i[2]
    train_data, valid_data, test_data = dataset_split(label_duration)
    np.savez_compressed('../dataset/label_duration.npz', train=train_data, valid=valid_data, test=test_data)

def create_dataset(db_cursor):
    if not os.path.isdir('../dataset/'):
        os.mkdir('../dataset/')
    create_info(db_cursor=db_cursor)
    create_disease(db_cursor=db_cursor)
    create_drug(db_cursor=db_cursor)
    create_duration(db_cursor=db_cursor)


if __name__ == "__main__":
    start = time.clock()
    connect = DBOptions.connect_db()
    cursor = connect.cursor()
    create_dataset(db_cursor=cursor)
    cursor.close()
    connect.close()
    print(time.clock() - start)
