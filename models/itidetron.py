"""
Filename: model.py
Author: Li Yi
Contact: liyi@kongfoo.cn
"""
import csv
import gc
import time
import random
import sys, json
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim 
import utils.evaluate as evaluate
import utils.functions as functions
import nnet.seqmodel as seqmodel
import utils.boxToCodon as boxToCodon
from random import randint
from dataset.codonDataset import CodonDataset 
from utils.codonToBox import CodonToBox 
from utils.download import GeneDownloader
from utils.language_helpers import seq_to_num 
from dataset.geneDataset import GeneDataset 
from dataset.codonFeat import CodonFeat, CodonFeatTransformer
from dataset.probioFeat import KmerSplitter
from models.model import Model
from models.fselect import FSelect 
from models.combiner import Combiner 
from models.classifier import Classifier
from models.genetica import Genetica
try:
    import cPickle as pickle
except ImportError:
    import pickle
import yaml
import pandas as pd
import os
import wandb

class ITidetron(Model):
    """
    Model类，用于初始化模型并进行训练或推理。
    
    Attributes:
        config_file (str): 模型配置文件路径。
        __version__ (str): 当前版本号。
        config (dict): 模型配置文件。
        status (str): 模型状态，train或predict。
        type (str): 模型类型，promoter、rbs或codon。
        train_dir (str): 训练集路径。
        dev_dir (str): 验证集路径。
        test_dir (str): 测试集路径。
        raw_file (str): 输入序列文件路径。
        decode_dir (str): 输出序列文件路径。
        optimizer (str): 优化器。
        loss_function (str): 损失函数。
        epochs (int): 训练轮数。
        batch_size (int): 训练批次大小。
        
    """
    def __init__(self, config_file, __version__):
        """
        Model类实例初始化。
        """
        super(Model, self).__init__()
        self.config_file = config_file
        self.config = None
        self.read_config()
        if wandb.run: 
            wandb.config.update(self.config)
            wandb.config.version = __version__
        logging.info(f"Running iTidetron platform with the following config...\n{json.dumps(self.config, indent=2)}")
        if self.status.lower()=='download':
            GeneDownloader(self.config['fna_dir'], self.config['raw_path'], self.config['ncores'])
        elif self.type.lower()=='codon':
            for path in [self.config['base_origin'], self.config['base_result']]:
                folder = os.path.exists(path)
                if not folder:                   
                    os.makedirs(path)
            self.data=CodonDataset(self.config)
            fna = GeneDataset(self.config)
            fna.load()
            fna.save()
            codonfeat=CodonFeat(self.config)
            codonfeat.load()
            codonfeat.save()
        elif self.type.lower()=='rbs' or self.type.lower()=='promoter':
            for path in [self.config['base_origin'], self.config['base_result']]:
                folder = os.path.exists(path)
                if not folder:                   
                    os.makedirs(path)
            self.data=CodonDataset(self.config)
            self.config['type']=self.type.lower()
            fna = GeneDataset(self.config)
            fna.load()
            fna.save()
            self.config['rna_score'] = None
            codonfeat=CodonFeat(self.config)
            codonfeat.load()
            codonfeat.save()
        elif self.type.lower() == 'probiotic':
            #生成数据储存文件夹（pipeline_data和final_result）
            for path in [self.config['result_path'], self.config['save_path']]:
                folder = os.path.exists(path)
                if not folder:                   
                    os.makedirs(path)
            self.config['raw_dir']=self.config['raw_path']
            del self.config['raw_path']
            if self.status.lower()=='train' and self.config['init_feat']:
                logging.info('Data Process Begins...')
                fna = GeneDataset(self.config)
                fna.main()
                fna.save()
                logging.info('Data Process Success!\n')
            elif self.status.lower()=='predict' and self.config['init_feat']:
                """
                数据处理部分。
                """
                #self.config.update(self.config.pop('predict_config'))
                logging.info('Data Process Begins...')
                fna = GeneDataset(self.config)# fna2pkl_predict
                fna.load()
                fna.save()
                logging.info('Data Process Success!\n')

    def train(self):
        """
        按照type的分类分别训练生菌或者密码子模型,
        """
        def _ifs():
            logging.info('Feature Selection Begins...')
            # feature
            fea = KmerSplitter(self.config)
            fea.load()
            fea.save()
            # fselect
            vars = self.k_mer_set  # [2, 3, 4, 5, 6, 7, 8]
            #assert set(vars).issubset(set(range(2, 9))), 'invalid k_mer_set'
            for k in vars:
                train_file = f"{self.config['save_path']}{k}{self.config['input_dir_fselect']}"
                dataset = pd.read_csv(f'{train_file}.csv')
                fsel = FSelect(train_file, dataset, k, self.config)
                sel_fv = fsel.train()
                sel_fv = pd.DataFrame(sel_fv)
                sel_fv.to_csv(f"{self.config['save_path']}{k}{self.config['output_dir_fselect']}", index=False)
            # spliter
            # k_mer_set = self.config['k_mer_set']  # [2, 3, 4, 5, 6, 7, 8]
            comb = Combiner(self.k_mer_set, self.config)
            comb.train()
            comb.save()
            logging.info('Feature Selection Success!\n')

        if self.type.lower()=='codon':
            """
            密码子模型训练部分。
            """
            logging.info('Expression Prediction Model Training Starts...\n')
            classi = Classifier(self.config)
            scores = classi.train()
            metrics = pd.DataFrame({
                "max_acc": [scores['test_accuracy'].max()],
                "avg_acc": [scores['test_accuracy'].mean()],
                "auc": [scores['test_roc_auc'].mean()]
            })
            if wandb.run:
                for i in range(len(scores)):
                    wandb.log({
                        "Expression Test acc":scores['pred_acc'],
                        "Expression Test auc":scores['pred_roc'],
                        "Expression Max val acc": [scores['test_accuracy'].max()],
                        "Expression Avg val acc": [scores['test_accuracy'].mean()],
                        "Expression Avg val roc": [scores['test_roc_auc'].mean()]
                    })
            logging.info('Expression Prediction Model Training Success!\n')
            classi.predict(genes = \
                ["GCF_902386435.1_UHGG_MGYG-HGUT-02372,ATGGATGCCCAACTAAATAATCTCTGGGAACAACAATTTCC",\
                "GCF_900637995.1_54466_G01,ATCTCGATCTTCGAGGGCCACTCCTCGGTCCGAGCGCGAGCGCCCCGTGTAGTAATTT"],\
                    ground_truth = [0,1], preprocesser = CodonFeatTransformer())
            self.codon2box()
            in_lines = open(self.train_dir,'r').readlines()
            self.data.build_alphabet(in_lines)
            self.data.generate_instance('train',in_lines)
            in_lines = open(self.dev_dir,'r').readlines()
            self.data.build_alphabet(in_lines)
            self.data.generate_instance('dev',in_lines)
            in_lines = open(self.test_dir,'r').readlines()
            self.data.build_alphabet(in_lines)
            self.data.generate_instance('test',in_lines)
            save_data_name = self.data.model_path +".dset"
            self.data.save(save_data_name)
            self.model = seqmodel.SeqModel(self.data)
            if self.data.optimizer.lower() == "sgd":
                optimizer = optim.SGD(self.model.parameters(), lr=self.data.HP_lr, momentum=self.data.HP_momentum,weight_decay=self.data.HP_l2)
            elif self.data.optimizer.lower() == "adagrad":
                optimizer = optim.Adagrad(self.model.parameters(), lr=self.data.HP_lr, weight_decay=self.data.HP_l2)
            elif self.data.optimizer.lower() == "adadelta":
                optimizer = optim.Adadelta(self.model.parameters(), lr=self.data.HP_lr, weight_decay=self.data.HP_l2)
            elif self.data.optimizer.lower() == "rmsprop":
                optimizer = optim.RMSprop(self.model.parameters(), lr=self.data.HP_lr, weight_decay=self.data.HP_l2)
            elif self.data.optimizer.lower() == "adam":
                optimizer = optim.Adam(self.model.parameters(), lr=self.data.HP_lr, weight_decay=self.data.HP_l2)
            else:
                logging.error("Optimizer illegal: %s"%(self.data.optimizer))
                exit(1)
            best_dev = -10
            # data.HP_iteration = 1
            ## start training
            # for idx in range(data.HP_iteration):
            for idx in range(self.epochs):
                print("Epoch: %s" %(idx))
                epoch_start = time.time()
                temp_start = epoch_start
                logging.info("Epoch: %s" %(idx))
                if self.data.optimizer == "SGD":
                        lr = self.data.HP_lr/(1+self.data.HP_lr_decay*idx)
                        logging.info(" Learning rate is setted as:", lr)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                instance_count = 0
                sample_id = 0
                sample_loss = 0
                total_loss = 0
                right_token = 0
                whole_token = 0
                expression_acc = 0
                random.shuffle(self.data.train_Ids)
                ## set model in train model
                self.model.train()
                self.model.zero_grad()
                batch_size = self.data.HP_batch_size
                batch_id = 0
                train_num = len(self.data.train_Ids)
                total_batch = train_num//batch_size+1
                for batch_id in range(total_batch):
                    start = batch_id*batch_size
                    end = (batch_id+1)*batch_size
                    if end >train_num:
                        end = train_num
                    instance = self.data.train_Ids[start:end]
                    if not instance:
                        continue
                    batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = functions.batchify_with_label(instance, self.data.HP_gpu)
                    instance_count += 1
                    expr_acc,loss, tag_seq = self.model.neg_log_likelihood_loss(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_wordrecover,batch_charrecover, batch_label, mask,classi)
                    right, whole = functions.predict_check(tag_seq, batch_label, mask)
                    right_token += right
                    whole_token += whole
                    expression_acc += expr_acc
                    sample_loss += loss.item()
                    total_loss += loss.item()
                    if end%500 == 0:
                        temp_time = time.time()
                        temp_cost = temp_time - temp_start
                        temp_start = temp_time
                        logging.info("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                        if sample_loss > 1e8 or str(sample_loss) == "nan":
                            logging.error("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                            exit(1)
                        sys.stdout.flush()
                        sample_loss = 0
                    loss.backward()
                    optimizer.step()
                    self.model.zero_grad()
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                expression_acc = expression_acc/total_batch
                logging.info("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f;expr_acc:%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token,expression_acc))
                epoch_finish = time.time()
                epoch_cost = epoch_finish - epoch_start
                logging.info("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
                logging.info("totalloss: %s" %total_loss)
                if wandb.run:
                    wandb.log({"train_Time":epoch_cost,"avg_train_acc":(right_token+0.)/whole_token,"total_loss":total_loss,"expr_avg_trian_acc":expression_acc})
                with open(self.loss_dir,'a') as f:
                    f.write(str(total_loss)+'\n')
                if total_loss > 1e8 or str(total_loss) == "nan":
                    logging.error("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                # continue
                # ## 训练集准确率
                speed, acc, p, r, f, _,_ ,expr_acc = evaluate.evaluate(self.data, self.model, "train", classi)
                train_finish = time.time()
                train_cost = train_finish - epoch_finish
                if self.data.seg:
                    logging.info("Train: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f,expr_acc:%.4f"%(train_cost, speed, acc, p, r, f,expr_acc))
                else:
                    logging.info("Train: time: %.2fs, speed: %.2fst/s; acc: %.4f,expr_acc:%.4f"%(train_cost, speed, acc,expr_acc))
                print("Train: time: %.2fs, speed: %.2fst/s; acc: %.4f,expr_acc:%.4f"%(train_cost, speed, acc,expr_acc))
                if wandb.run:
                    wandb.log({"train_eval_Time":train_cost,"train_acc":(right_token+0.)/whole_token,"train_expr_acc":expr_acc})
                speed, acc, p, r, f, _,_ ,expr_acc= evaluate.evaluate(self.data, self.model,  "dev", classi)
                dev_finish = time.time()
                dev_cost = dev_finish - train_finish
                if self.data.seg:
                    current_score = f
                    logging.info("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f,expr_acc:%.4f"%(dev_cost, speed, acc, p, r, f,expr_acc))
                else:
                    current_score = acc
                    logging.info("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f,expr_acc:%.4f"%(dev_cost, speed, acc,expr_acc))
                if wandb.run:
                    wandb.log({"dev_eval_Time":dev_cost,"dev_acc":acc,"dev_expr_acc":expr_acc})
                if current_score > best_dev:
                    if self.data.seg:
                        logging.info("Exceed previous best f score:%.2f"%best_dev)
                    else:
                        logging.info("Exceed previous best acc score:%.2f"%best_dev)
                    model_name = self.data.model_path +'.'+ str(idx) + ".model"
                    logging.info("Save current best model in file:%s"%model_name)
                    torch.save(self.model.state_dict(), model_name)
                    best_dev = current_score

                # ## decode test
                speed, acc, p, r, f, _,_ ,expr_acc= evaluate.evaluate(self.data, self.model,  "test", classi)
                test_finish = time.time()
                test_cost = test_finish - dev_finish
                if self.data.seg:
                    logging.info("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f,expr_acc:%.4f"%(test_cost, speed, acc, p, r, f,expr_acc))
                else:
                    logging.info("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f,expr_acc:%.4f"%(test_cost, speed, acc,expr_acc))
                print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f,expr_acc:%.4f"%(test_cost, speed, acc,expr_acc))
                if wandb.run:
                    wandb.log({"test_eval_Time":test_cost,"test_acc":acc,"test_expr_acc":expr_acc})
                gc.collect()

        elif self.type.lower()=='rbs' or self.type.lower()=='promoter':
            """
            RBS/promoter训练部分。
            """
            logging.info('Expression Prediction Model Training Starts...\n')
            classi = Classifier(self.config)
            scores = classi.train()
            metrics = pd.DataFrame({
                "max_acc": [scores['test_accuracy'].max()],
                "avg_acc": [scores['test_accuracy'].mean()],
                "auc": [scores['test_roc_auc'].mean()]
            })
            if wandb.run:
                for i in range(len(scores)):
                    wandb.log({
                        "Expression Test acc":scores['pred_acc'],
                        "Expression Test auc":scores['pred_roc'],
                        "Expression Max val acc": [scores['test_accuracy'].max()],
                        "Expression Avg val acc": [scores['test_accuracy'].mean()],
                        "Expression Avg val roc": [scores['test_roc_auc'].mean()]
                    })
            logging.info('Expression Prediction Model Training Success!\n')
            self.initial_population_path=self.config["initial_population_path"]
            with open(self.initial_population_path) as f:
                seqs=f.readlines()
            f.close()
            self.initial_populations=[]
            accuracy = 0.0
            pred,acc = classi.predict(seqs)
            accuracy -= sum(pred==self.expected_result)
            logging.info('Before mutation,promoter expression accuracy is %.4f' %(-accuracy/len(seqs)))
            for seq in seqs:
                seq = seq_to_num(seq.replace('\n', ''))
                chromosome = []
                for i in range(len(seq)):
                    chromosome.append(seq[i])
                self.initial_populations.append(chromosome)
            if self.type.lower()=='rbs':
                accuracy = 0.0
                seqs_mutated=[]
                for initial_population in self.initial_populations:
                    genetica = Genetica(self.config,classi,[initial_population for _ in range(self.config['num_parents_mating'])])
                    accuracy -= genetica.main()
                    logging.info('After mutation,codon expression accuracy is %.4f' %(-accuracy/len(seqs)))
                    seqs_mutated.append(genetica.save_solution)
                for seq in seqs_mutated:
                    genetica = Genetica(self.config,classi,codon_part_sequence=seq)
                    accuracy += genetica.main()
                    genetica.train()
                accuracy /= len(self.initial_populations)
                logging.info('After mutation,RBS expression accuracy improved %.4f' %accuracy)
                return
            else:
                for initial_population in self.initial_populations:
                    genetica = Genetica(self.config,classi,[initial_population for _ in range(self.config['num_parents_mating'])])
                    accuracy += genetica.main()
                    genetica.train()
            accuracy /= len(self.initial_populations)
            logging.info('After mutation,promoter expression accuracy improved %.4f' %accuracy)

        elif self.type.lower()=='probiotic':
            """
            益生菌模型训练部分。
            """
            if self.config['re_ifs']: _ifs()
            logging.info('Model Train Begins...')
            # classifier
            classi = Classifier(self.config)
            scores = classi.train()
            if wandb.run:
                for i in range(len(scores)):
                    wandb.log({
                        "Test acc":scores['pred_acc'],
                        "Test auc":scores['pred_roc'],
                        "Max val acc": [scores['test_accuracy'].max()],
                        "Avg val acc": [scores['test_accuracy'].mean()],
                        "Avg val roc": [scores['test_roc_auc'].mean()]
                    })
            logging.info('Model Train Success!\n')

    def predict(self):
        """
        按照type的分类分别加载益生菌或者密码子模型，完成推理得到结果。
        """
        if self.type.lower()=='codon':
            logging.info("MODEL: predict")
            self.data.load(self.data.dset_dir)
            self.data.read_config(self.config)
            f=open(self.raw_path,'r')
            input_str=f.read()
            f.close()
            count, err, s_len, output=CodonToBox(input_str=input_str)
            f=open(self.raw_file,'w')
            f.write(output)
            f.close()
            # exit(0)
            self.data.show_data_summary()
            in_lines = open(self.raw_file,'r').readlines()
            self.data.generate_instance('raw',in_lines)
            logging.info("nbest: %s"%(self.data.nbest))
            classi = Classifier(self.config)
            model = seqmodel.SeqModel(self.data)
            model.load_state_dict(torch.load(self.data.load_model_dir, map_location='cpu'))
            _, _, _, _, _, decode_results, pred_scores,_ = evaluate.evaluate(self.data, model, 'raw',classi, self.data.nbest)
            
            if self.data.nbest:
                self.data.write_nbest_decoded_results(decode_results, pred_scores, 'raw', classi)
            else:
                self.data.write_decoded_results(decode_results, 'raw')

        elif self.type.lower()=='probiotic':
            """
            模型推理部分。
            """
            classi = Classifier(self.config)
            if self.config["sample_genes"]:
                data = pd.read_csv(self.config["raw_dir"],skiprows=[0,1], quoting=csv.QUOTE_ALL,header=None)
                genes = data.values.tolist()
                genes = [item for gene in genes for item in gene]
                y_hat, pred_acc = classi.predict(genes=genes,ground_truth="use_config")
            else:
                if self.config['normalize']:
                    if self.config['do_splitting']:
                        fea = KmerSplitter(self.config)
                        fea.predict()
                    comb = Combiner(self.k_mer_set, self.config)
                    feats, _ = comb.predict()
                # print("here")
                # exit()
                y_hat, pred_acc = classi.predict(feats=feats, ground_truth=self.config['ground_truth'], output_proba = True)
            logging.info(f"Predicted labels: {y_hat}")
            logging.info(f"Predicted accuracy: {pred_acc}")
                
    def read_config(self):
        """
        读取配置文件。
        """
        with open(self.config_file) as f:#将yaml文件写为字典形式
            self.config = yaml.safe_load(f)
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
            #表达水平和RNA结构分数的原始数据
            'raw_dir': 'raw_dir',
            'labels_dir': 'labels_dir',
            'initial_population_path': 'initial_population_path',
            'rna_score': 'rna_score',
            #标注模型预测时加载之前保存的训练好的模型参数
            'load_model_dir': 'load_model_dir',
            #表达水平模型预测fna2pkl生成的特征数据
            'save_dir_feat': 'save_dir_feat',
            # #fna2pkl配置
            'k_mer_set': 'k_mer_set',
            'ncores': 'ncores',
            #codonFeat将序列数据与标签的特征划分训练集测试集后输出结果
            'fine_feats': 'fine_feats',
            'fine_feat_train': 'fine_feat_train',
            'fine_feat_test': 'fine_feat_test',
            'labels':'labels',
            'labels_train': 'labels_train',
            'labels_test': 'labels_test',
            #classifier类模型训练结果分数
            'model_dir' : 'model_dir',
            'save_metrics': 'save_metrics',
            'save_scores': 'save_scores',
            'save_core_select': 'save_core_select',
            #classfier类参数
            'verbose': 'verbose',
            'num_feats': 'num_feats',
            'num_folds': 'num_folds',
            'njobs': 'njobs',
            'scaler': 'scaler',
            'test_ratio': 'test_ratio',
            'model_name' : 'model_name',
            'svm__C': 'svm__C',
            'svm__kernel' : 'svm__kernel',
            'svm__gamma': 'svm__gamma',
            'xgb__max_depth': 'xgb__max_depth',
            'xgb__n_estimators': 'xgb__n_estimators',
                    }
        for item, attr in attr_mapping.items():
            if item in self.config:
                setattr(self, attr, self.config[item])
        self.config.update(self.config.pop(f'{self.status.lower()}_config'))
        attr_mapping = {   
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
            #密码子标注模型预测时生成的优化后序列个数
            'nbest': 'nbest',
            #promoter&RBS预期结果
            'expected_result':'expected_result'
            }
        for item, attr in attr_mapping.items():
            if item in self.config:
                setattr(self, attr, self.config[item])

    def codon2box(self):
        """
        将密码子序列转换为密码子对序列。
        """
        paths = [self.train_path, self.dev_path, self.test_path]
        dirs = [self.train_dir, self.dev_dir, self.test_dir]
        
        for path, output_dir in zip(paths, dirs):
            with open(path, 'r') as f:
                input_str = f.read()
            
            count, err, s_len, output = CodonToBox(input_str=input_str)
            
            with open(output_dir, 'w') as f:
                f.write(output)
