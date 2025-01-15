"""
Filename: fna2pkl.py
Author: Wang Chuyu
Contact: wangchuyu@kongfoo.cn
"""
import logging
import os
import pandas as pd
import multiprocessing
from itertools import islice
from dataset.dataset import Dataset
from utils.language_helpers import NgramLM

class GeneDataset(Dataset):
    """
    读取原始基因序列数据集，计算子序列，作为基因组特征。
    """
    def __init__(self, config):
        self.read_config(config)
        self.config = config
        self.raw_data = self.read_dataframe()
        self.k_mer_set = config['k_mer_set']

    def read_config(self, config):
        """
        读取配置文件。
        """
        self.raw_dir = config['raw_dir']
        self.ncores = config['ncores']
        self.save_dir = config['save_dir_feat']
        self.k_mer_set = config['k_mer_set']

    def read_dataframe(self):
        """
        读取原始数据。
        """
        raw_data = pd.read_csv(self.raw_dir)
        try:
            assert raw_data.shape[1] == 1, f'{self.raw_dir} does not have one column.'
            assert raw_data.iloc[0][0] == 'names,feature', f'{self.raw_dir} does not have the right header.'
        except AssertionError as ae:
            logging.error(ae)
        return raw_data

    def save(self):
        """
        保存输出结果。
        """
        root_dir=os.path.dirname(os.path.abspath(self.save_dir))
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.genomic_feat.to_csv(self.save_dir, index=True)

    def process_row(self, row):
        """
        计算每个子序列在样本中出现的频率。

        参数:
            row: 菌株名称和碱基序列。
        返回:
            已处理子序列的频率。

        示例:
        >>> process_row(row)
        """
        mers2num = {}
        for it_index in self.mers:
            mers2num[it_index] = 0
        seq = row[1][0].split(',')[-1]
        name = row[1][0].split(',')[0]
        for i in range(len(seq)):
            for j in self.k_mer_set:
                if i + j <= len(seq) and seq[i:i+j] in mers2num:
                    mers2num[seq[i:i+j]] = mers2num[seq[i:i+j]] + 1
        feats = []
        for i, mer in enumerate(self.mers):
            num = mers2num[mer]
            feats.append(num / (len(seq) + 1 - len(mer)))
        return name, feats

    def main(self):
        """
        生成长度为k(2-8)的所有可能的基因序列，
        然后计算每个子序列在样本中出现的频率。
        """
        self.mers = []
        for i in self.k_mer_set:
            buf = NgramLM(i,[]).kmers() #self.dfs('', i)
            self.mers.extend(buf)
        # logging.info("mers:", len(self.mers))

        # 初始化多进程，使用所需进程数量的进程池。
        # 根据需要更改进程数量。
        pool = multiprocessing.Pool(processes=self.ncores)

        # 将raw_data分割成要并行处理的行。
        rows = islice(self.raw_data.iterrows(), 1, None)

        # 使用pool.map对每一行并行应用process_row函数。
        results = pool.map(self.process_row, rows)

        # 关闭pool，释放资源。
        pool.close()
        pool.join()

        # 将结果存储在genomic_feat字典中。
        self.genomic_feat = pd.DataFrame({name: feats for name, feats in results if feats is not None})

    def load(self):
        self.main()
        self.genomic_feat['feat_name'] = self.mers
        self.genomic_feat = self.genomic_feat.set_index('feat_name')

# 调用方法
# if __name__ == "__main__":
    # appendix = dataiku.get_custom_variables()["appendix"]
    # assert appendix in set(["651", "114", "239"]), 'invalid dataset appendix'
    # raw_data = dataiku.Dataset(f'Probiotic{appendix}')
    # raw_data = pd.DataFrame(raw_data.get_dataframe())
    # ncores = dataiku.get_custom_variables()["ncores"]  # ncores = 64

    # fna2pkl = Fna2Pkl(raw_data, ncores)
    # genomic_feat = fna2pkl.main()

    # feat = dataiku.Dataset(f'feat_train_{appendix}')
    # feat.write_with_schema(pd.DataFrame(data=genomic_feat))
