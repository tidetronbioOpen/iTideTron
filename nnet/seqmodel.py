"""
Filename: seqmodel.py
Author: Li Yi
Contact: liyi@kongfoo.cn
"""

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnet.wordsequence import WordSequence
from nnet.crf import CRF
from models.classifier import Classifier
from utils.boxToCodon import box_to_codon
from utils.functions import recover_label


class SeqModel(nn.Module):
    """
    SeqModel类,作为密码子部分的训练模型
    继承了基类Model类
    同时引用了WordSequence类以及CRF类实例及函数等
    """
    def __init__(self, data):
        """
        构造函数，输入参数为dataset类实例
        """
        super(SeqModel, self).__init__()
        self.use_crf = data.use_crf
        print("build network...")
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)
        print("use crf: ", self.use_crf)
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        self.crf_ranking_ratio = data.crf_ranking_ratio
        self.expected_result = data.expected_result
        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.word_hidden = WordSequence(data)
        self.word_alphabet = data.word_alphabet
        self.label_alphabet = data.label_alphabet
        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

    def neg_log_likelihood_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,batch_wordrecover, char_seq_recover, batch_label, mask,classifier):
        """
        计算损失函数
        参数:
        - 多个输入特征张量
        返回:
        - total_loss: 损失值
        - tag_seq: 解码的序列
        """
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        batch_ori_seq=self.batch_boxtocodon(word_inputs,batch_label,mask,batch_wordrecover)
        ori_expression_results=expression_pred(batch_ori_seq,classifier)
        ori_expr_acc=1-get_ranking_loss_binary_classification(ori_expression_results,self.expected_result)/word_inputs.size(0)
        batch_seq=self.batch_boxtocodon(word_inputs,tag_seq,mask,batch_wordrecover)
        expression_results=expression_pred(batch_seq,classifier)
        ranking_loss=get_ranking_loss_binary_classification(expression_results,self.expected_result)
        total_loss= self.crf_ranking_ratio*total_loss+ranking_loss
        expr_acc=1-get_ranking_loss_binary_classification(expression_results,self.expected_result)/word_inputs.size(0)
        expr_acc=expr_acc-ori_expr_acc
        if self.average_batch:
            total_loss = total_loss / batch_size
        return expr_acc,total_loss, tag_seq

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        """
        前向计算
        参数:
        - 多个输入特征张量
        返回:        
        - tag_seq: 解码得到的序列
        """
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq  = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq
        return tag_seq

    # def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)

    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest):
        """
        解码Top N最优结果
        
        参数:
        - 多个输入特征张量  
        - nbest: Top N
        
        返回:
        - scores: 每个解码序列的分数
        - tag_seq: Top N解码序列
        
        """
        if not self.use_crf:
            print("Nbest output is currently supported only for CRF! Exit...")
            exit(0)
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq
    
    def batch_boxtocodon(self,batch_word,batch_label,mask,batch_wordrecover):
        batch_word = batch_word[batch_wordrecover]
        batch_label = batch_label[batch_wordrecover]
        mask = mask[batch_wordrecover]
        mask = mask.cpu().data.numpy()
        batch_word_numpy=batch_word.cpu().numpy()
        batch_label_numpy=batch_label.cpu().numpy()
        assert batch_word_numpy.shape==batch_label_numpy.shape==mask.shape
        batch_size = batch_word.size(0)
        seq_len = batch_word.size(1)
        seqs=[]
        for i in range(batch_size):
            word_id=batch_word_numpy[i]
            label_id=batch_label_numpy[i]
            sequence=''
            for j in range(seq_len):
                if mask[i][j]==0:
                    continue
                word=self.word_alphabet.get_instance(word_id[j])
                label=self.label_alphabet.get_instance(label_id[j])
                sequence += word + ' ' + label + '\n'
            count, err_count,s_len,output_str=box_to_codon(sequence)
            seqs.append(output_str)
        return seqs

def get_ranking_loss_binary_classification(expression_results,expected_result):
    ranking_loss=0
    for result in expression_results:
        if result==0 and expected_result==1:
            ranking_loss+=1
        elif result==1 and expected_result==0:
            ranking_loss+=1
        elif result==expected_result==0 or result==expected_result==1:
            continue
    return ranking_loss

def get_ranking_loss_triple_class_classification(expression_results,expected_result):
    ranking_loss=0
    for result in expression_results:
        if result==0 and expected_result==1:
            ranking_loss+=1
        elif result==1 and expected_result==0:
            ranking_loss+=1
        elif result==2 and expected_result==0 or result==2 and expected_result==1:
            ranking_loss+=0.5
    return ranking_loss

def get_ranking_loss_softmax_output(expression_results,expected_result):
    ranking_loss=0
    a=[[0,1,0.5],[1,0,0.5]]
    assert expected_result==0 or expected_result==1
    ratio=a[expected_result]
    for result in expression_results:
        for i in range(3):
            ranking_loss+=result[i]*ratio[i]
    return ranking_loss

      
def expression_pred(batch_seq,classifier):
    result,_=classifier.predict(batch_seq)
    return result
