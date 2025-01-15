"""
Filename: codondataset.py
Author: Li Yi
Contact: liyi@kongfoo.cn
"""
from __future__ import print_function
from __future__ import absolute_import
import sys

from utils.functions import *
from utils.alphabet import Alphabet
from utils.boxToCodon import box_to_codon
from utils.functions import str2bool
from dataset.dataset import Dataset
import torch 
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle
from dataset.codonFeat import CodonFeatTransformer

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"

class CodonDataset(Dataset):
    """
    Data类，继承了dataset类，作为密码子优化部分的数据类
    在dataset类的基础上增加了以下函数功能：
    read_config函数读取配置文件
    show_data_summary打印数据信息
    build_alphabet由序列数据创建字母表
    """
    def __init__(self,config):
        """
        初始化Data类实例
        """
        super(CodonDataset,self).__init__(config)
        self.MAX_SENTENCE_LENGTH =500
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.feature_name = []
        self.feature_alphabets = []
        self.feature_num = len(self.feature_alphabets)
        self.feat_config = None
        self.label_alphabet = Alphabet('label',True)
        self.tagScheme = "NoSeg" ## BMES/BIO

        self.seg = True

        ### I/O
        self.train_dir = None
        self.dev_dir = None
        self.test_dir = None
        self.raw_file = None

        self.decode_dir = None
        self.dset_dir = None ## data vocabulary related file
        self.model_path = None ## model save  file
        self.load_model_dir = None ## model load file

        self.word_emb_dir = None
        self.char_emb_dir = None
        self.feature_emb_dirs = []

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.feature_alphabet_sizes = []
        self.feature_emb_dims = []
        self.norm_feature_embs = []
        self.word_emb_dim = 50
        self.char_emb_dim = 30

        ###Networks
        self.word_feature_extractor = "LSTM" ## "LSTM"/"CNN"/"GRU"/
        self.use_char = True
        self.char_feature_extractor = "LSTM" ## "LSTM"/"CNN"/"GRU"/None
        self.use_crf = True
        self.nbest = None

        ## Training
        self.expected_result = 1
        self.crf_ranking_ratio = 0.2
        self.average_batch_loss = False
        self.optimizer = "SGD" ## "SGD"/"AdaGrad"/"AdaDelta"/"RMSProp"/"Adam"
        self.status = "train"
        ### Hyperparameters
        self.HP_cnn_layer = 4
        self.HP_iteration = 10000
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8
        self.read_config(config)

    def show_data_summary(self):
        """
        打印数据信息
        """
        print("++"*50)
        print("DATA SUMMARY START:")
        print(" I/O:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding  dir: %s"%(self.word_emb_dir))
        print("     Char embedding  dir: %s"%(self.char_emb_dir))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Norm   word     emb: %s"%(self.norm_word_emb))
        print("     Norm   char     emb: %s"%(self.norm_char_emb))
        print("     Train  file directory: %s"%(self.train_dir))
        print("     Dev    file directory: %s"%(self.dev_dir))
        print("     Test   file directory: %s"%(self.test_dir))
        print("     Raw    file directory: %s"%(self.raw_file))
        print("     Dset   file directory: %s"%(self.dset_dir))
        print("     Model  file directory: %s"%(self.model_path))
        print("     Loadmodel   directory: %s"%(self.load_model_dir))
        print("     Decode file directory: %s"%(self.decode_dir))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Raw   instance number: %s"%(len(self.raw_texts)))
        print("     FEATURE num: %s"%(self.feature_num))
        # for idx in range(self.feature_num):
        #     print("         Fe: %s  alphabet  size: %s"%(self.feature_alphabets[idx].name, self.feature_alphabet_sizes[idx]))
        #     print("         Fe: %s  embedding  dir: %s"%(self.feature_alphabets[idx].name, self.feature_emb_dirs[idx]))
        #     print("         Fe: %s  embedding size: %s"%(self.feature_alphabets[idx].name, self.feature_emb_dims[idx]))
        #     print("         Fe: %s  norm       emb: %s"%(self.feature_alphabets[idx].name, self.norm_feature_embs[idx]))
        print(" "+"++"*20)
        print(" Model Network:")
        print("     Model        use_crf: %s"%(self.use_crf))
        print("     Model word extractor: %s"%(self.word_feature_extractor))
        print("     Model       use_char: %s"%(self.use_char))
        if self.use_char:
            print("     Model char extractor: %s"%(self.char_feature_extractor))
            print("     Model char_hidden_dim: %s"%(self.HP_char_hidden_dim))
        print(" "+"++"*20)
        print(" Training:")
        print("     Optimizer: %s"%(self.optimizer))
        print("     Iteration: %s"%(self.HP_iteration))
        print("     BatchSize: %s"%(self.HP_batch_size))
        print("     Average  batch   loss: %s"%(self.average_batch_loss))

        print(" "+"++"*20)
        print(" Hyperparameters:")

        print("     Hyper              lr: %s"%(self.HP_lr))
        print("     Hyper        lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyper         HP_clip: %s"%(self.HP_clip))
        print("     Hyper        momentum: %s"%(self.HP_momentum))
        print("     Hyper              l2: %s"%(self.HP_l2))
        print("     Hyper      hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyper         dropout: %s"%(self.HP_dropout))
        print("     Hyper      lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyper          bilstm: %s"%(self.HP_bilstm))
        print("     Hyper             GPU: %s"%(self.HP_gpu))
        print("DATA SUMMARY END.")
        print("++"*50)
        sys.stdout.flush()

    def build_alphabet(self, in_lines):
        '''
        根据数据序列创建字母表
        字母表与字典类似，实际是创建序列中所有元素的集合，
        并建立元素与对应key值的映射
        '''
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                ## build feature alphabet
                # for idx in range(self.feature_num):
                #     feat_idx = pairs[idx+1].split(']',1)[-1]
                #     self.feature_alphabets[idx].add(feat_idx)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        # for idx in range(self.feature_num):
        #     self.feature_alphabet_sizes[idx] = self.feature_alphabets[idx].size()
        # startS = False
        # startB = False
        # for label,_ in self.label_alphabet.iteritems():
        #     if "S-" in label.upper():
        #         startS = True
        #     elif "B-" in label.upper():
        #         startB = True
        # if startB:
        #     if startS:
        #         self.tagScheme = "BMES"
        #     else:
        #         self.tagScheme = "BIO"


    def fix_alphabet(self):
        '''
        关闭字母表，固定字母表的内容
        '''        
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        # for idx in range(self.feature_num):
        #     self.feature_alphabets[idx].close()


    def build_pretrain_emb(self):
        '''
        如果配置文件中给出了预训练嵌入的路径，生成预训练嵌入
        '''  
        # if self.word_emb_dir:
        #     print("Load pretrained word embedding, norm: %s, dir: %s"%(self.norm_word_emb, self.word_emb_dir))
        #     self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)
        # if self.char_emb_dir:
        #     print("Load pretrained char embedding, norm: %s, dir: %s"%(self.norm_char_emb, self.char_emb_dir))
        #     self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(self.char_emb_dir, self.char_alphabet, self.char_emb_dim, self.norm_char_emb)
        # for idx in range(self.feature_num):
        #     if self.feature_emb_dirs[idx]:
        #         print("Load pretrained feature %s embedding:, norm: %s, dir: %s"%(self.feature_name[idx], self.norm_feature_embs[idx], self.feature_emb_dirs[idx]))
        #         self.pretrain_feature_embeddings[idx], self.feature_emb_dims[idx] = build_pretrain_embedding(self.feature_emb_dirs[idx], self.feature_alphabets[idx], self.feature_emb_dims[idx], self.norm_feature_embs[idx])


    def generate_instance(self, name,in_lines):

        '''
        如果配置文件中给出了预训练嵌入的路径，生成预训练嵌入
        '''  
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(in_lines, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(in_lines, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(in_lines, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance(in_lines, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    def write_decoded_results(self, predict_results, name):
        """
        将预测（标注）完毕的优化密码子序列写入输出文件
        """
        fout = open(self.decode_dir,'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, self.decode_dir))


    def load(self,data_file):
        """
        从文件路径加载数据内容
        """
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self,save_file):
        """
        向输出文件路径保存数据内容
        """
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def write_nbest_decoded_results(self, predict_results, pred_scores, name,classifier):
        """
        将n个最好的预测（标注）完毕的优化密码子序列写入输出文件
        """
        ## predict_results : [whole_sent_num, nbest, each_sent_length]
        ## pred_scores: [whole_sent_num, nbest]
        fout = open(self.decode_dir,'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        assert(sent_num == len(pred_scores))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx][0])
            nbest = len(predict_results[idx])
            score_string = "# "
            for idz in range(nbest):
                score_string += format(pred_scores[idx][idz], '.4f')+" "
            fout.write(score_string.strip() + "\n")
            codon_list = []
            for idy in range(nbest):
                box_string=""
                for idz in range(sent_length):
                    box_string += content_list[idx][0][idz] + " "+ predict_results[idx][idy][idz]+ '\n'
                count, err_count, seq_len, codon_string = box_to_codon(box_string)
                codon_list.append(codon_string)
            
            y_proba,_ = classifier.predict(genes=codon_list,ground_truth=[1 for i in range(len(codon_list))],preprocesser=CodonFeatTransformer(),output_proba=True)
            prob_score_list = []
            for probas in y_proba:
                prob_score_list.append(probas[1])
            prob_scores = dict(zip(codon_list,prob_score_list))
            sorted_codon = sorted(prob_scores.items(), key=lambda x: x[1],reverse=True)
            top10 = sorted_codon[:10]
            
            for idy in range(len(top10)):
                optimized_codon_sequence=self.decode_path+".txt"
                with open(optimized_codon_sequence, 'w') as f:
                    for item in top10:
                        string = item[0]
                        number = item[1]
                        f.write(string + '\t' + str(number) + '\n')
                f.close()
                
            for idy in range(sent_length):
                try:  # Will fail with python3
                    label_string = content_list[idx][0][idy].encode('utf-8') + " "
                except:
                    label_string = content_list[idx][0][idy] + " "
                for idz in range(nbest):
                    label_string += predict_results[idx][idz][idy]+ " "
                label_string = label_string.strip() + "\n"
                fout.write(label_string)
            fout.write('\n')
        fout.close()
        print("Predict %s %s-best result has been written into file. %s"%(name,nbest, self.decode_dir))

    def read_config(self,config):
        """
        读取配置文件
        """
        self.config=config
        # read data:
        attr_mapping = {
            #一些公共配置
            'type': 'type',
            'status': 'status',
            'expected_result': 'expected_result',
            #用于训练标注模型的原始数据
            # base_origin: data/codon/origin_data
            # base_inter: data/codon/inter_data
            # base_result: data/codon/result_data
            'train_path': 'train_path',
            'dev_path': 'dev_path',
            'test_path': 'test_path',   
            #密码子标注模型训练模块参数
            #密码子序列经过codon2box得到的氨基酸密码子对数据
            'train_dir': 'train_dir',
            'dev_dir': 'dev_dir',
            'test_dir': 'test_dir',
            #模型训练过程中的loss与模型参数
            'model_path': 'model_path',
            'loss_dir': 'loss_dir',
            #预训练词嵌入配置
            'word_emb_dir': 'word_emb_dir',
            #训练过程中的超参数配置
            'epochs': 'epochs',
            'learning_rate': 'learning_rate',
            'lr_decay': 'lr_decay',
            'momentum': 'momentum',
            'l2': 'l2',
            'optimizer': 'optimizer',
            'iteration': 'iteration',
            'batch_size': 'batch_size',
            'gpu': 'gpu',
            #seqmodel类各种超参数
            'crf_ranking_ratio': 'crf_ranking_ratio',
            'char_hidden_dim': 'char_hidden_dim',
            'hidden_dim': 'hidden_dim',
            'dropout': 'dropout',
            'lstm_layer': 'lstm_layer',
            'bilstm': 'bilstm',
            'cnn_layer': 'cnn_layer',
            'word_emb_dim': 'word_emb_dim',
            'char_emb_dim': 'char_emb_dim',
            'word_seq_feature': 'word_seq_feature',
            'char_seq_feature': 'char_seq_feature',
            'use_crf': 'use_crf',
            'use_char': 'use_char',
            'ave_batch_loss': 'ave_batch_loss',
            'norm_word_emb': 'norm_word_emb',
            'norm_char_emb': 'norm_char_emb',
            'number_normalized': 'number_normalized',
            'seg': 'seg',
            #密码子标注模型预测模块参数：
            #待优化的密码子序列
            'raw_path': 'raw_path',
            #经过codon2box得到的氨基酸密码子对数据
            'raw_file': 'raw_file',
            #优化过后的氨基酸密码子对序列
            'decode_dir': 'decode_dir',
            #优化过后的密码子序列
            'decode_path': 'decode_path',
            #密码子标注模型预测时加载codonDataset数据
            'dset_dir': 'dset_dir',
            #标注模型预测时加载之前保存的训练好的模型参数
            'load_model_dir': 'load_model_dir',
            #密码子标注模型预测时生成的优化后序列个数
            'nbest': 'nbest'
            }
        for item, attr in attr_mapping.items():
            if item in config:
                setattr(self, attr, self.config[item])
        attr_mapping = {
            'use_char': 'use_char',
            'use_crf': 'use_crf',
            'gpu': 'gpu',
            'ave_batch_loss': 'ave_batch_loss',
            'norm_char_emb': 'norm_char_emb',
            'norm_word_emb': 'norm_word_emb',
            'number_normalized': 'number_normalized',
            'seg': 'seg',

        }
        for item, attr in attr_mapping.items():
            if item in config:
                setattr(self, attr, str2bool(self.config[item]))
        self.HP_gpu =  torch.cuda.is_available() and self.gpu

