"""
Filename: feature.py
Author: Wang Chuyu
Contact: wangchuyu@kongfoo.cn
"""
import csv, json, sys
import logging
import random
import math
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
from dataset.dataset import Dataset
from utils.language_helpers import NgramLM
from sklearn.model_selection import StratifiedShuffleSplit

class KmerSplitter(Dataset):
    """
    将输入特征根据基因中不同长度的子序列频率分成八个部分，
    将每部分划分为训练集和测试集，并写入文件。
    """
    def __init__(self, config):
        self.read_config(config)
        self.config = config
        self.read_data()

    def read_config(self, config):
        """
        读取配置文件。
        """
        self.feat_dir = config['save_dir_feat']
        self.save_dir = config['save_path']
        self.k_mer_set = config['k_mer_set']
        if config['status'] == 'train':
            self.labels_dir = config['labels_dir']
            self.save_dir_train = config['save_dir_feature_train']
            self.save_dir_test = config['save_dir_feature_test']

    def read_data(self):
        """
        读取原始数据。
        """
        def _process_row(row, header, offset):
            # Iterate through each row in the CSV
            for column_name, value in zip(header, row[offset:]):
                self.feat_data[column_name].append(value)

        self.feat_data = {}
        with open(self.feat_dir, newline='') as file:
            reader = csv.reader(file)
            # Read the header to get column names
            header = next(reader)
            assert len(header) > 1, 'invalid header length'
            offset = 0
            if header[0] == 'ID' or 'feat_name':
                # ignore the first column if it is the index
                header = header[1:] 
                offset = 1
            # Initialize the dictionary with empty lists for each column
            for column_name in header:
                self.feat_data[column_name] = []
            # Create a ThreadPoolExecutor with ncores of workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['ncores']) as executor:
                # Read lines in parallel and process them
                futures = [executor.submit(_process_row, row, header, offset) for row in reader]
                # Wait for all tasks to complete
                concurrent.futures.wait(futures)

    def save(self):
        """
        保存输出结果。
        """
        for i, k in enumerate(self.k_mer_set):
            header = NgramLM(k,[]).kmers()
            assert len(header) == self.train_data[i].shape[1] - 1, 'invalid header length'
            assert len(header) == self.test_data[i].shape[1] - 1, 'invalid header length'
            train_data_save = pd.DataFrame(self.train_data[i])
            train_data_save.columns = ['label'] + header
            filled_cols = train_data_save.iloc[:, 1:].min() if self.config['column_fill_zero_with'] == 'min'\
                else train_data_save.iloc[0, 1:] - train_data_save.iloc[0, 1:] + sys.float_info.epsilon
            train_data_save = train_data_save.apply(lambda col: col.replace(0, filled_cols[col.name])
                                                        if col.name in filled_cols else col)
            train_data_save.to_csv(f'{self.save_dir}{k}{self.save_dir_train}', index=False)
            test_data_save = pd.DataFrame(self.test_data[i])
            test_data_save.columns = ['label'] + header
            filled_cols = test_data_save.iloc[:, 1:].min() if self.config['column_fill_zero_with'] == 'min'\
                else test_data_save.iloc[0, 1:] - test_data_save.iloc[0, 1:] + sys.float_info.epsilon
            test_data_save = test_data_save.apply(lambda col: col.replace(0, filled_cols[col.name])
                                                        if col.name in filled_cols else col)
            test_data_save.to_csv(f'{self.save_dir}{k}{self.save_dir_test}', index=False)

    def writedata(self, samples, labels):
        """
        将样本数据和标签写入文件。

        参数:
            samples(list): 样本数据的字典列表，每个字典包含样本的特征信息。
            labels(list): 标签列表，对应每个样本的标签。
            filename(str): 输出文件名。

        示例:
            samples = [{'feature1': 0.5, 'feature2': 1.2}, {'feature1': 0.1, 'feature2': 0.8}]
            labels = [1, 0]
            writedata(samples, labels, 'output.txt')
        """
        num = len(samples)

        if len(samples) == 0:
            return None

        num_y = len(samples[0])
        if labels == "predict":
            data = np.zeros((num, (num_y)))
        else:
            data = np.zeros((num, (num_y+1)))

        for i in range(num):
            if labels:
                try:
                    data[i][0] = labels[i]
                except:
                    logging.info(f"Labels {i} in {num} not found, data shape: {len(data)}, {len(data[0])}")
            else:
                data[i][0] = 0
                logging.info(f"Labels not provided: {i}, {num}, {len(data)}, {len(data[0])}")
            kk = list(samples[i].keys())
            kk.sort()
            for k in kk:
                try:
                    if labels == "predict":
                        data[i][k] = samples[i][k]
                    else:
                        data[i][k+1] = samples[i][k]
                except:
                    logging.debug(f"Shifting feature exception: {data[i][k+1]}, {type(data[i][k+1])}")

        return data

    def load(self):
        """
        读取碱基序列的标签数据和特征信息，生成不同长度的k-mer子序列，
        随机划分数据集为训练集和测试集，并将样本数据和标签写入文件。
        """
        labels = json.load(open(self.labels_dir))
        #labels = self.labels_data
        feats = self.feat_data 
        
        names = list(feats)
        my_labels = [labels[my_name] for my_name in names]
        my_seed = 1
        random.seed(my_seed)
        split = StratifiedShuffleSplit(n_splits=1, test_size=self.config['test_ratio'], random_state=my_seed)
        for train_index, test_index in split.split(names, my_labels):
            test_names = np.array(names)[test_index]
        # get names
        train_names = [k for k in feats.keys() if k not in test_names]
        names_json = []
        for k in train_names:
            names_json.append(k)
        # for k in test_names:
        #     names_json.append(k)
        names_json = {k:-1 for k in names_json}
        with open("my_train_names_json.json","w") as f:
            f.write(json.dumps(names_json))
        # exit()
        # names = list(feats)
        # random.shuffle(names)
        # test_names = names[0:math.ceil(len(names) * self.config['test_ratio'])]

        mers = []
        k_mer = []
        for i in self.k_mer_set:
            buf = NgramLM(i,[]).kmers() #self.dfs('', i)
            k = [i for _ in buf]
            mers.extend(buf)
            k_mer.extend(k)

        # assert set(self.k_mer_set).issubset(range(2, 9)), 'invalid k_mer set'

        self.train_data = []
        self.test_data = []
        for i in tqdm(self.k_mer_set):
            ids = []
            for index, k in enumerate(k_mer):
                if k == i:
                    ids.append(index)
            ids = np.array(ids)
            train_samples, test_samples, train_labels, test_labels = [], [], [], []
            for name, feat in feats.items():
                # logging.info(f'Processing {name}')
                feat = feat[ids[0]:int(ids[-1]+1)]
                # assert len(feat) == len(ids)
                val = {}
                for k, item in enumerate(feat):
                    val[k] = item
                if name not in test_names:
                    train_labels.append(labels[name])
                    train_samples.append(val)
                else:
                    test_labels.append(labels[name])
                    test_samples.append(val)

            self.train_data.append(self.writedata(train_samples, train_labels))
            self.test_data.append(self.writedata(test_samples, test_labels))

    def predict(self):
        # feat = dataiku.Dataset("feat_predict")
        # feat = feat.get_dataframe()
        feats = self.feat_data 

        mers = []
        k_mer = []
        for i in self.k_mer_set:
            buf = NgramLM(i,[]).kmers() #self.dfs('', i)
            k = [i for _ in buf]
            mers.extend(buf)
            k_mer.extend(k)

        # assert set(self.k_mer_set).issubset(range(2, 9)), 'invalid k_mer set'
        
        if max(feats) == 'feat_name':
            feats = dict(list(feats.items())[1:])
            
        for i in self.k_mer_set:
            ids = []
            for index, k in enumerate(k_mer):
                if k == i:
                    ids.append(index)
            ids = np.array(ids)
            samples = []
            index = []
            for name, feat in feats.items():
                # logging.info(f'Processing {name}')
                feat = feat[ids[0]:int(ids[-1]+1)]
                # assert len(feat) == len(ids)
                val = {}
                for k, item in enumerate(feat):
                    val[k] = item
                index.append(name)
                samples.append(val)
            predict_data = pd.DataFrame(self.writedata(samples, "predict"))
            predict_data.columns = NgramLM(i,[]).kmers()
            predict_data['ID'] = index
            predict_data.set_index('ID', inplace=True)
            filled_cols = predict_data.iloc[:, 1:].min() if self.config['column_fill_zero_with'] == 'min'\
                else predict_data.iloc[0, 1:] - predict_data.iloc[0, 1:] + sys.float_info.epsilon
            predict_data = predict_data.apply(lambda col: col.replace(0, filled_cols[col.name])
                                                        if col.name in filled_cols else col)
            # test_data_save.to_csv(f'{self.save_dir}{k}{self.save_dir_test}', index=False)
            predict_data.to_csv(f"{self.config['save_path']}{i}{self.config['save_dir_feature_predict']}", index=True)

# 调用方法
# if __name__ == "__main__":
    # appendix = dataiku.get_custom_variables()["appendix"]
    # assert appendix in set(["651","114","239"]), 'invalid dataset appendix'
    # labels = dataiku.Dataset(f"labels{appendix}")
    # labels = labels.get_dataframe()
    # feat = dataiku.Dataset(f"feat_train_{appendix}")
    # feat = feat.get_dataframe()
    # k_mer_set = dataiku.get_custom_variables()["k_mer_set"]

    # feature = Feature(labels, feat, k_mer_set)
    # train_data, test_data = feature.main()

    # k_mer_list = json.loads(k_mer_set)
    # for i in k_mer_list:
    #     dataset = dataiku.Dataset(f'{i}_mer_train_file_')
    #     dataset.write_with_schema(pd.DataFrame(data=train_data))
    #     dataset = dataiku.Dataset(f'{i}_mer_test_file_')
    #     dataset.write_with_schema(pd.DataFrame(data=test_data))
