"""
Filename: spliter.py
Author: Wang Chuyu
Contact: wangchuyu@kongfoo.cn
"""
import logging
import pandas as pd
from tqdm import tqdm
from models.model import Model
import numpy as np
np.random.seed(1024)

class Combiner(Model):
    """
    主要功能：针对fselect分别筛选得到的不同kmer各自的特征子集，
    选取原始特征矩阵中的对应列，并将不同k的结果合并为一个特征集。
    """
    def __init__(self, k_mer_set, config):
        self.k_mer_set = k_mer_set
        self.config = config
        self.X_train = []
        self.X_test = []
        self.X_predict = []
        self.y_train = []
        self.y_test = []

    def read_sample(self, line):
        """
        数据标准化。

        参数：
        - line：碱基序列。

        返回：
        计算结果。

        示例：
        >>> read_sample(line[1])
        """
        point = {}
        for key, value in enumerate(line):
            point[key] = float(value)
        factor = 1.0 / sum(point.values())
        if self.config['do_norm_combiner']:
            point = {k: v * factor for k, v in point.items()}
        return {k: v * factor for k, v in point.items()}, len(point)

    def read_selected_features(self, k):
        """
        读取选择的特征ID。

        参数：
        - k：子序列长度。

        返回：
        计算结果。

        示例：
        >>> read_selected_features(k)
        """
        # v2_mer_train_select = dataiku.Dataset(f'{k}_mer_train_select_')  # change
        # lines = v2_mer_train_select.get_dataframe()
        v2_mer_train_select = pd.read_csv(f"{self.config['save_path']}{k}{self.config['output_dir_fselect']}")
        lines = v2_mer_train_select.values.tolist()
        feature_id = [int(line[0]) for line in lines]
        return feature_id

    def read_selected_samples(self, k, selected_feats_idx, tot, offset_feat):
        """
        读取选择的特征。

        参数：
        - k：子序列长度。
        - selected_feats_idx：选择的特征ID。
        - tot：训练集或测试集。
        - offset_feat：偏移特征。

        返回：
        选择的特征和标签。

        示例：
        >>> read_selected_samples(k, selected_feats_idx, 'train', offset_feat)
        """
        # file = dataiku.Dataset(f'{k}_mer_{tot}_file_')  # change
        # lines = file.get_dataframe()
        file = pd.read_csv(f"{self.config['save_path']}{k}_mer_{tot}_file.csv")
        lines = file.astype(str)
        if not tot == "predict":
            labels = lines['label'].tolist()
            del lines['label']
            logging.debug(lines)
        if tot == "predict":
            IDs = lines['ID'].tolist()
            del lines['ID']
            logging.debug(lines)
        self.kmers = lines.columns.tolist()
        selected_samples = []
        for line in lines.iterrows():
            sample, dim = self.read_sample(line[1])
            selected_feats = {}
            for key in sample.keys():
                if int(key) in selected_feats_idx:
                    offset_feat[f'{k}_{key}'] = offset_feat.get(f'{k}_{key}', len(offset_feat)+1)
                    selected_feats[key] = sample[key]
                    # selected_feats[offset_feat[f'{k}_{key}']] = sample[key]
            selected_feats = dict(sorted(selected_feats.items(),key=lambda x: selected_feats_idx.index(x[0])))
            selected_idx = list(offset_feat.values())
            selected_feats = {selected_idx[i]:v for i,v in enumerate(selected_feats.values())}
            selected_samples.append(' '.join(f'{k}:{v}' for k,v in selected_feats.items()))

        if not tot == "predict":
            return selected_samples, labels, offset_feat.copy()
        else:
            return selected_samples, IDs, None 

    def mask_by_embeddings(self,samples):
        """
        根据kmer特征嵌入对训练数据集使用dropout方法进行增强
        Args:
            seld (_type_): _description_
            samples (_type_): _description_

        Returns:
            samples: 经过dropout方法增强后的训练数据集
        """
        if self.config['kmer_emb']:
            kmer_emb = pd.read_pickle(self.config['kmer_emb'])
            zero_feat = kmer_emb.apply(lambda a: np.absolute(np.array(a).mean().round(2))).to_frame()
            zero_feat = zero_feat.T
            zero_probs = zero_feat.loc[:,samples.columns].iloc[0]/self.config['dropout_factor']
            zero_counts = [int(len(samples)*zero_prob) for zero_prob in zero_probs]
            indices = [np.random.choice(samples.index,count) for count in zero_counts]
            for i,idx in enumerate(indices):
                samples.iloc[idx,i] = 0
            return samples
        return samples
    
    def train(self):
        """
        首先从自定义变量中读取k_mer_set，然后对每个k进行特征选择和数据划分。
        通过调用read_selected_features函数读取选定的特征ID，
        然后通过read_selected_samples函数读取选定的特征和标签。
        将训练集和测试集的特征进行组合，并保存到fine_feat_train_和fine_feat_test_两个数据集中，
        将训练集和测试集的标签保存到labels_train_和labels_test_中，
        并将所有的特征和标签组合，保存到fine_feats_和labels_两个数据集中。
        """
        selected_train_feats = []
        selected_test_feats = []
        train_labels = None
        test_labels = None
        offset_feat = {}
        # vars = json.loads(dataiku.get_custom_variables()["k_mer_set"])
        vars = self.k_mer_set
        # assert set(vars).issubset(range(2, 9)), 'invalid {k}_mer set'
        self.header = []
        for k in vars:  # 窗口大小
            selected_feats_idx = self.read_selected_features(k)
            selected_train_feat, train_labels, offset_train = self.read_selected_samples(k, selected_feats_idx, 'train', offset_feat)
            selected_test_feat, test_labels, offset_test = self.read_selected_samples(k, selected_feats_idx, 'test', offset_feat)
            self.header.extend([self.kmers[i] for i in selected_feats_idx])
            assert(len(train_labels) == len(selected_train_feat))
            assert(len(test_labels) == len(selected_test_feat))
            assert(offset_train == offset_test)
            selected_train_feats.append(selected_train_feat)
            selected_test_feats.append(selected_test_feat)
            offset_feat = offset_test

        # 将每个k-mer的选择特征组合在一起，并将它们保存到名为all_train_file.txt的文件中
        feats_num = len(selected_train_feats[0])
        feats_size = len(selected_train_feats)
        train_data = [[] * (feats_size + 1)] * feats_num
        for i in tqdm(range(feats_num)):
            train_data[i].append(train_labels[i])
            self.y_train.append(float(train_labels[i]))
            line = []
            for j in range(feats_size):
                train_data[i].append(selected_train_feats[j][i])
                line.extend([float(feat.split(':')[1]) for feat
                             in selected_train_feats[j][i].split(' ')])
            self.X_train.append(line)
            logging.debug(f'处理第 {i} 行')

        feats_num = len(selected_test_feats[0])
        feats_size = len(selected_test_feats)
        test_data = [[] * (feats_size + 1)] * feats_num

        # 将每个k-mer的选择特征组合在一起，并将它们保存到名为all_train_file.txt的文件中
        for i in tqdm(range(len(selected_test_feats[0]))):
            test_data[i].append(test_labels[i])
            self.y_test.append(float(test_labels[i]))
            line = []
            leng = len(selected_test_feats)
            for j in range(leng):
                test_data[i].append(selected_test_feats[j][i])
                line.extend([float(feat.split(':')[1]) for feat
                             in selected_test_feats[j][i].split(' ')])
            self.X_test.append(line)
            logging.debug(f'处理第 {i} 行')

    def predict(self):
        selected_train_feats = []
        offset_feat = {} 
        self.header = []
        for k in self.k_mer_set:  # 窗口大小
            selected_feats_idx = self.read_selected_features(k)
            logging.debug(offset_feat)
            selected_train_feat, IDs , _ = self.read_selected_samples(k, selected_feats_idx, 'predict', offset_feat)
            self.header.extend([self.kmers[i] for i in selected_feats_idx])
            selected_train_feats.append(selected_train_feat)

        feats_num = len(selected_train_feats[0])
        feats_size = len(selected_train_feats)
        train_data = [[] * feats_size] * feats_num
        for i in range(feats_num):
            line = []
            for j in range(feats_size):
                train_data[i].append(selected_train_feats[j][i])
                line.extend([float(feat.split(':')[1]) for feat in selected_train_feats[j][i].split(' ')])
            self.X_predict.append(line)
            logging.debug(f'processing row {i}')

        self.X_predict = pd.DataFrame(self.X_predict)
        self.X_predict.columns = self.header
        self.X_predict['ID'] = IDs
        self.X_predict.set_index('ID', inplace=True)
        self.X_predict.to_csv(self.config['fine_feat_predict'], index=True)
        return self.X_predict, self.header
    
    def save(self):
        fine_feat_train = pd.DataFrame(self.X_train)
        fine_feat_train.columns = self.header
        fine_feat_train = self.mask_by_embeddings(fine_feat_train)
        fine_feat_train.to_csv(self.config['fine_feat_train'], index=False)
        fine_feat_test = pd.DataFrame(self.X_test)
        fine_feat_test.columns = self.header
        fine_feat_test.to_csv(self.config['fine_feat_test'], index=False)
        fine_feats = pd.DataFrame(self.X_train + self.X_test)
        fine_feats.columns = self.header
        fine_feats.to_csv(self.config['fine_feats'], index=False)
        labels_train = pd.DataFrame(self.y_train)
        labels_train.to_csv(self.config['labels_train'], index=False)
        labels_test = pd.DataFrame(self.y_test)
        labels_test.to_csv(self.config['labels_test'], index=False)
        labels = pd.DataFrame(self.y_train + self.y_test)
        labels.to_csv(self.config['labels'], index=False)

# 调用方法
# if __name__ == "__main__":
#     # k_mer_set = dataiku.get_custom_variables()["k_mer_set"]
#     k_mer_set = [2, 3, 4, 5, 6, 7, 8]

#     spliter = Spliter(k_mer_set)
#     X_train, X_test, y_train, y_test = spliter.main()

#     fine_feat_train = pd.DataFrame(X_train)
#     fine_feat_train.to_csv('test/fine_feat_train.csv', index=False)

#     fine_feat_test = pd.DataFrame(X_test)
#     fine_feat_test.to_csv('test/fine_feat_test.csv', index=False)

#     fine_feats = pd.DataFrame(X_train+X_test)
#     fine_feats.to_csv('test/fine_feats.csv', index=False)

#     labels_train = pd.DataFrame(y_train)
#     labels_train.to_csv('test/labels_train.csv', index=False)

#     labels_test = pd.DataFrame(y_test)
#     labels_test.to_csv('test/labels_test.csv', index=False)

#     labels = pd.DataFrame(y_train+y_test)
#     labels.to_csv('test/labels.csv', index=False)
