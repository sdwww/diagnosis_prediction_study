from data_process import CreateDataset

def get_config():
    model_config = dict()
    # 模型参数
    model_config['n_disease'] = CreateDataset.disease_num
    model_config['n_drug'] = CreateDataset.drug_num
    model_config['n_visit'] = CreateDataset.visit_num
    model_config['n_disease_category'] = CreateDataset.disease_category_num
    model_config['n_drug_category'] = CreateDataset.drug_category_num
    model_config['n_embed'] = 600
    model_config['n_rnn'] = (300, 300)
    model_config['dropout_rate'] = 0.5
    model_config['predict_drug'] = True
    # 数据集路径
    model_config['disease_file'] = '../dataset/dataset_disease.npz'
    model_config['disease_label'] = '../dataset/label_disease_categ.npz'
    model_config['drug_file'] = '../dataset/dataset_drug.npz'
    model_config['drug_label'] = '../dataset/label_drug_categ.npz'
    # 训练参数
    model_config['out_file'] = '../attention_model_result/result'
    model_config['model_file'] = '../attention_model_result/'
    model_config['n_epoch'] = 1
    model_config['batch_size'] = 500
    model_config['save_max_keep'] = 1
    return model_config