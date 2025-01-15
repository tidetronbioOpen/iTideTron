"""
Filename: codonFeat.py
Author: Li Yi
Contact: liyi@kongfoo.cn
"""
import os
import pandas as pd
from dataset.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

class CodonFeatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #for input feature,drop None and redundant features
        data_df = X.transpose()
        # rna_score_df=pd.read_csv(self.rna_score,header=0,index_col=0)
        # data_df = data_df.join(rna_score_df)
        # data_df=data_df.rename(columns={'rna_struct_score': data_df.shape[1]-1})
        data_df = data_df.dropna(axis=1, how='all')
        unique_counts = data_df.nunique()
        redundant_feats = unique_counts[unique_counts == 1].index
        data_df.drop(redundant_feats, axis=1, inplace=True)
        return data_df
    
class CodonFeat(Dataset):
    """
    主要功能是对样本进行去重以及训练集验证集测试集划分
    """
    def __init__(self,config):
        """
        Dataset类实例初始化
        """
        self.read_config(config)

    def load(self,data_file=None):
        """
        从data_file路径加载数据创建数据集
        """
        assert os.path.exists(self.save_dir_feat)==True
        self.raw_data = pd.read_csv(self.save_dir_feat)
        self.data = CodonFeatTransformer().fit_transform(self.raw_data)
        self.data.columns = self.data.iloc[0]
        self.data = self.data[1:]
        self.data.reset_index(drop=True, inplace=True)
        self.train_feat, self.test_feat = train_test_split(self.data, test_size=self.test_ratio, random_state=1)
        assert os.path.exists(self.labels_dir)==True
        self.label_df = pd.read_csv(self.labels_dir)
        self.label_df = self.label_df.transpose()
        unique_counts = self.label_df.nunique()
        redundant_feats = unique_counts[unique_counts == 1].index
        self.label_df.drop(redundant_feats, axis=1, inplace=True)
        self.label_df = self.label_df.dropna(axis=1, how='all')
        self.train_label, self.test_label = train_test_split(self.label_df, test_size=self.test_ratio, random_state=1)

    def save(self,save_file=None):
        """
        将数据保存在save_file路径下
        """
        if save_file: 
            root_dir = os.path.dirname(os.path.abspath(save_file))
            if not os.path.exists(root_dir): os.makedirs(root_dir)       
            self.data.to_csv(save_file,index=False)
        else:
            self.data.to_csv(self.fine_feats,index=False)
        self.train_feat.to_csv(self.fine_feat_train,index=False)
        self.test_feat.to_csv(self.fine_feat_test,index=False)
        self.label_df.to_csv(self.labels,index=False)
        self.train_label.to_csv(self.labels_train,index=False)
        self.test_label.to_csv(self.labels_test,index=False)

    def read_config(self,config):
        self.save_dir_feat=config['save_dir_feat']
        self.labels_dir=config['labels_dir']
        self.fine_feats=config['fine_feats']
        self.fine_feat_train=config['fine_feat_train']
        self.fine_feat_test=config['fine_feat_test']
        self.labels=config['labels']
        self.labels_train=config['labels_train']
        self.labels_test=config['labels_test']
        self.test_ratio=config['test_ratio']
        self.rna_score=config['rna_score']
