"""
Filename: fselect.py
Author: Wang Chuyu
Contact: wangchuyu@kongfoo.cn
"""
import os
import sys
import json
import random
import logging, pickle
import pandas as pd
from models.model import Model
from datetime import datetime
from random import randrange
from subprocess import *

# pylint: disable = W0311

# logging.info(os.getcwd())
# current_dir = os.getcwd()
# parent_dir = os.path.dirname(current_dir)
# parent_dir2 = os.path.dirname(parent_dir)
# parent_dir3 = os.path.dirname(parent_dir2)
# parent_dir4 = os.path.dirname(parent_dir3)
# os.chdir(parent_dir4)

current_path = os.path.dirname(os.path.abspath(__file__))
main_folder_path = os.path.join(current_path, '..')

is_win32 = (sys.platform == 'win32')
if not is_win32:
    gridpy_exe = f"python {main_folder_path}/utils/grid.py -log2c -2,9,2 -log2g 1,-11,-2"
    svmtrain_exe = "./svm-train"
    svmpredict_exe = "./svm-predict"
else:
    gridpy_exe = rf"python {main_folder_path}\utils\grid.py -log2c -2,9,2 -log2g 1,-11,-2"
    svmtrain_exe = r"..\windows\svmtrain.exe"
    svmpredict_exe = r"..\windows\svmpredict.exe"

##### 全局变量 #####

# train_pathfile = ""
# train_file = ""
# test_pathfile = ""
# test_file = ""
# if_predict_all = 0

# whole_fsc_dict = {}
# whole_imp_v = []
class FSelect(Model):
    """
    基于数据集train_file，通过F-score方法选择最优特征数量，并将选择后的特征数据写入文件。
    """
    def __init__(self, train_file, dataset, k, config):
        self.config = config
        self.train_file = train_file
        self.dataset = dataset
        self.k = k
        self.kmers = []
        self.EPSILON = 1e-12 

        logging.info(f"----------- START FEATURE SELECION FOR {k}_MER ----------")

    def writedata(self, samples, labels, filename):
        """
        将数据写入到文件中。
        """
        fp = sys.stdout
        if filename:
            fp = open(filename, "w")

        num = len(samples)
        for i in range(num):
            if labels:
                fp.write("%s" % labels[i])
            else:
                fp.write("0")
            kk = list(samples[i].keys())
            kk.sort()
            for k in kk:
                fp.write(" %d:%f" % (k, samples[i][k]))
            fp.write("\n")

        fp.flush()
        fp.close()

    def feat_num_try_half(self, max_index):
        """
        尝试选取特征数量的一半。
        """
        v = []
        while max_index > 1:
            v.append(max_index)
            max_index //= 2
        return v

    def feat_num_try(self, f_tuple):
        """
        尝试选取特征数量。
        """
        LENGTH = len(f_tuple)
        for i in range(LENGTH):
            if f_tuple[i][1] < 1e-20:  # 1e-20 f-score
                logging.debug(f'Break at {i}!')
                i = i - 1
                break
        # 只取后八个数（大于1%的特征）
        return self.feat_num_try_half(i + 1)[-8:]

    def cal_feat_imp(self, label, sample):
        """
        计算特征的重要性。
        """
        logging.debug("Calculating fsc...")
        score_dict = self.cal_Fscore(label, sample)

        score_tuples = list(score_dict.items())
        score_tuples.sort(key=self.value_cmpf)

        feat_v = score_tuples
        LENGTH = len(feat_v)
        for i in range(LENGTH):
            feat_v[i] = score_tuples[i][0]

        logging.debug("Fsc done.")
        return score_dict, feat_v

    def readdata(self):
        """
        读取数据。
        """
        dataset = self.dataset.astype(str)
        labels = dataset['label'].tolist()
        del dataset['label']
        self.kmers = dataset.columns.tolist()
        max_index = dataset.shape[1]
        dataset = dataset.values.tolist()
        samples = []
        for i, line in enumerate(dataset):
            sample = {}
            for j, item in enumerate(line):
                sample[j] = float(item)
            samples.append(sample)
        return labels, samples, max_index

    def random_shuffle(self, label, sample):
        """
        随机洗牌。
        """
        random.seed(1)  # 使结果每次相同
        size = len(label)
        for i in range(size):
            ri = randrange(0, size - i)
            tmp = label[ri]
            label[ri] = label[size - i - 1]
            label[size - i - 1] = tmp
            tmp = sample[ri]
            sample[ri] = sample[size - i - 1]
            sample[size - i - 1] = tmp

    def value_cmpf(self, _x):
        """
        用于对特征的重要性进行排序的比较函数。
        """
        return -_x[1]

    def cal_Fscore(self, labels, samples):
        """
        计算F-score。
        """
        data_num = float(len(samples))
        p_num = {} #key: label;  value: data num
        sum_f = [] #index: feat_idx;  value: sum
        sum_l_f = {} #dict of lists.  key1: label; index2: feat_idx; value: sum
        sumq_l_f = {} #dict of lists.  key1: label; index2: feat_idx; value: sum of square
        F = {} #key: feat_idx;  valud: fscore
        max_idx = -1

        ### 第一遍：计算每个类别的数量和特征的最大索引
        LENGTH = len(samples)
        for p in range(LENGTH): # 遍历每个数据点
            label = labels[p]
            point = samples[p]
            if label in p_num:
                p_num[label] += 1
            else:
                p_num[label] = 1

            for f in point.keys(): # 遍历每个特征
                if f > max_idx:
                    max_idx = f
        ### 现在 p_num 和 max_idx 被设置

        ### 初始化变量
        sum_f = [0 for i in range(max_idx)]
        for la in p_num.keys():
            sum_l_f[la] = [0 for i in range(max_idx)]
            sumq_l_f[la] = [0 for i in range(max_idx)]

        ### 第二遍：计算数据的一些统计数据
        LENGTH = len(samples)
        for p in range(LENGTH): # 遍历每个数据点
            point = samples[p]
            label = labels[p]
            for tuple in point.items(): # 遍历每个特征
                f = tuple[0] - 1 # 特征索引
                v = tuple[1] # 特征值
                sum_f[f] += v
                sum_l_f[label][f] += v
                sumq_l_f[label][f] += v ** 2
        ### 现在 sum_f、sum_l_f 和 sumq_l_f 完成

        ### 对每个特征计算F-score
        #eps = 1e-12
        for f in range(max_idx):
            SB = 0
            for la in p_num.keys():
                SB += (p_num[la] * (sum_l_f[la][f] / p_num[la] - sum_f[f] / data_num) ** 2)

            SW = self.EPSILON 
            for la in p_num.keys():
                SW += (sumq_l_f[la][f] - (sum_l_f[la][f] ** 2) / p_num[la])

            F[f + 1] = SB / SW

        return F

    def train_svm(self, tr_file):
        """
        使用训练数据训练SVM模型，并选择最佳参数。
        """
        cmd = "%s %s" % (gridpy_exe, tr_file)
        logging.debug(cmd)
        logging.debug('Cross validation...')
        std_out = Popen(cmd, shell=True, stdout=PIPE).stdout
        logging.debug(std_out)

        line = ''
        while 1:
            last_line = line
            logging.debug(last_line)
            line = std_out.readline()
            if not line:
                break
        logging.debug(type(last_line))
        c, g, rate = map(float, last_line.split())

        logging.debug('Best c=%s, g=%s CV rate=%s' % (c, g, rate))

        return c, g, rate

    def select(self, sample, feat_v):
        """
        选择指定的特征。
        """
        new_samp = []

        feat_v.sort()

        # 对于每个样本
        for s in sample:
            point = {}
            # 对于要选择的每个特征
            for f in feat_v:
                if f in s:
                    point[f] = s[f]
            factor = 1.0 / sum(point.values()) #+ sys.float_info.epsilon)
            if self.config['do_norm']:
                point = {k: v * factor for k, v in point.items()}
            new_samp.append(point)

        return new_samp

    def train(self):
        """
        从dataiku定义的自定义变量中读取k_mer_set，然后依次对每个k进行特征选择：
        根据给定的k构建相应的训练数据集train_file，
        读取数据，并对数据进行随机洗牌，确保数据的随机性。
        计算特征的重要性，得到特征排序列表whole_fsc_dict和whole_imp_v，
        使用F-score结果，尝试不同数量的特征，计算对应的交叉验证准确率。
        最后选择最佳的特征数量，并将对应的特征数据写入文件。
        """
        # train_file = f'{k}_mer_train_file_'
        labels, samples, max_index = self.readdata()
        self.random_shuffle(labels, samples)
        whole_fsc_dict, whole_imp_v = self.cal_feat_imp(labels, samples)
        #--------------------------------------
        # 保存F-score和重要度
        f_score_path = self.train_file + ".fscore"
        imp_score_path = self.train_file + ".imp"
        with open(f_score_path, 'wb') as f:
            pickle.dump(whole_fsc_dict, f)
        with open(imp_score_path, 'wb') as f:
            pickle.dump(whole_imp_v, f)
        #--------------------------------------
        # logging.info(whole_fsc_dict)

        f_tuples = list(whole_fsc_dict.items())
        f_tuples.sort(key=self.value_cmpf)
        # logging.info(f_tuples)

        accuracy = []
        fnum_v = self.feat_num_try(f_tuples)  # ex: [50,25,12,6,3,1]
        for i in range(len(fnum_v)):
            accuracy.append([])
        est_acc = []

        logging.info("Try feature sizes: %s\n" % (fnum_v))
        logging.info("Feat\test.\tacc.")
        acc = {}
        LENGTH = len(fnum_v)
        for j in range(LENGTH):
            fn = fnum_v[j]  # fn is the number of features selected
            fv = whole_imp_v[:fn]  # fv is indices of selected features

            # 选择特征
            tr_sel_samp = self.select(samples, fv)
            tr_sel_name = self.train_file + ".tr"
            self.writedata(tr_sel_samp, labels, tr_sel_name)

            # 从拆分的训练样本中选择最佳的C和Gamma值
            c, g, cv_acc = self.train_svm(tr_sel_name)
            est_acc.append(cv_acc)
            logging.info("%d:\t%.5f" % (fnum_v[j], cv_acc))
            acc[fn] = cv_acc

        fnum = fnum_v[est_acc.index(max(est_acc))]
        logging.info("Max validation accuarcy: %.6f\n" % max(est_acc))
        acc["max_acc"] = max(est_acc)
        json.dump(acc, open(f"{self.config['save_path']}{self.k}{self.config['json_dir_fselect']}", 'w'), ensure_ascii=False)

        sel_fv = whole_imp_v[:fnum]
        logging.info("Select features: %s"%sel_fv)
        logging.info("Selected kmers: %s\n"%' '.join([self.kmers[i] for i in sel_fv]))
        logging.info('Number of selected features %s\n' % fnum)

        return [{'idx':i,'kmer':self.kmers[i]} for i in sel_fv]
        # dataset = dataiku.Dataset(f"{k}_mer_train_select_")
        # dataset.write_with_schema(pd.DataFrame(sel_fv))

# 调用方法
# if __name__ == "__main__":
#     # vars = json.loads(dataiku.get_custom_variables()["k_mer_set"])
#     vars = [2, 3, 4, 5, 6, 7, 8]
#     assert set(vars).issubset(set(range(2, 9))), 'invalid k_mer_set'
#     # logging.info("start: %s\n\n" % datetime.now())

#     for k in vars:
#         train_file = f'test/{k}_mer_train_file'
#         # data = dataiku.Dataset(train_file)
#         # dataset = data.get_dataframe()
#         dataset = pd.read_csv(f'{train_file}.csv')

#         fselect = FSelect(dataset, k)
#         sel_fv = fselect.main()

#         # data = dataiku.Dataset(f"{k}_mer_train_select_")
#         # data.write_with_schema(pd.DataFrame(sel_fv))
#         sel_fv = pd.DataFrame(sel_fv)
#         sel_fv.to_csv(f"test/{k}_mer_train_select.csv", index=False)
