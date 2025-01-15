# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-06-10 17:49:50
from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np
import torch
import torch.autograd as autograd


def normalize_word(word):
    '''
    将字符串中的数字替换为0
    '''
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(in_lines, word_alphabet, char_alphabet, feature_alphabets, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    '''
    函数功能：从输入文本文件中读取序列标注实例
    输入参数:
      in_lines: 输入文件行列表
      word_alphabet: 单词Alphabet
      char_alphabet: 字符Alphabet  
      feature_alphabets:特征Alphabet列表
      label_alphabet: 标签Alphabet
      number_normalized: 是否数字规范化 
      max_sent_length: 最大句子长度
      char_padding_size: 字符padding大小  
      char_padding_symbol: 填充符号
    输出:
      instence_texts: 读取的文本实例  
      instence_Ids: 实例中各元素的id
    
    步骤:
      1. 读取每行,分词并获取标签
      2. 获取单词、字符、特征的id
      3. 如果达到最大长度或者文件结束,生成一个文本实例
    '''    
    feature_num = len(feature_alphabets)
    # print(label_alphabet.instance2index)
    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []
    labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    label_Ids = []
    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()

            if sys.version_info[0] < 3:
                word = pairs[0].decode('utf-8')
            else:
                word = pairs[0]

            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            label_Ids.append(label_alphabet.get_index(label))
            ## get features
            feat_list = []
            feat_Id = []
            for idx in range(feature_num):
                feat_idx = pairs[idx+1].split(']',1)[-1]
                feat_list.append(feat_idx)
                feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
            features.append(feat_list)
            feature_Ids.append(feat_Id)
            ## get char
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
            # if (len(words) > 0) and ((max_sent_length < 0)):
                instence_texts.append([words, features, chars, labels])
                instence_Ids.append([word_Ids, feature_Ids, char_Ids,label_Ids])
            words = []
            features = []
            chars = []
            labels = []
            word_Ids = []
            feature_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids


def write_instance(instence_texts, instence_Ids,texts_path,Ids_path):
    '''
    将instence_texts和instence_Ids写入texts_path和Ids_path
    '''
    # f=open(texts_path,'w')
    # for text in instence_texts[0]:
    #     f.writelines(str(text))
    #     f.writelines('\n')
    # f.close()
    # f=open(Ids_path,'w')
    # for Id in instence_Ids[0]:
    #     f.writelines(str(Id))
    #     f.writelines('\n')

    # f.close() 


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    '''
    从embedding_path加载预训练词嵌入向量，并进行归一化，返回预训练词嵌入的词数以及嵌入的维度
    '''
    # # 初始化字典
    # embedd_dict = dict()
    # # 如果embedding_path不为空
    # if embedding_path!= None:
    #     # 加载预训练词向量
    #     embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    # # 获取字典大小
    # alphabet_size = word_alphabet.size()
    # # 初始化预训练词向量
    # scale = np.sqrt(3.0 / embedd_dim)
    # pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    # # 完整匹配数
    # perfect_match = 0
    # # 忽略大小写匹配
    # case_match = 0
    # # 未匹配数
    # not_match = 0
    # # 遍历字典
    # for word, index in word_alphabet.iteritems():
    #     # 如果字典中存在该词
    #     if word in embedd_dict:
    #         # 如果当前词的词向量不为0
    #         if norm:
    #             # 计算当前词的归一化向量
    #             pretrain_emb[index,:] = norm2one(embedd_dict[word])
    #         else:
    #             # 计算当前词的原始向量
    #             pretrain_emb[index,:] = embedd_dict[word]
    #         # 完整匹配数加1
    #         perfect_match += 1
    #     # 如果字典中不存在该词
    #     elif word.lower() in embedd_dict:
    #         # 如果当前词的词向量不为0
    #         if norm:
    #             # 计算当前词的归一化向量
    #             pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
    #         else:
    #             # 计算当前词的原始向量
    #             pretrain_emb[index,:] = embedd_dict[word.lower()]
    #         # 忽略大小写匹配数加1
    #         case_match += 1
    #     else:
    #         # 如果当前词的词向量为0
    #         pretrain_emb[index-1,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
    #         # 未匹配数加1
    #         not_match += 1
    # # 计算预训练词向量的大小
    # pretrained_size = len(embedd_dict)
    # # 打印预训练词向量的大小
    # print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    # # 返回预训练词向量和词向量维度
    # return pretrain_emb, embedd_dim

def norm2one(vec):
    '''
    将向量归一化
    :param vec: 向量
    :return: 归一化之后的向量
    '''
    # root_sum_square = np.sqrt(np.sum(np.square(vec)))
    # return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    '''
    加载预训练的词向量
    :param embedding_path: 词向量路径
    :return: 词向量字典和词向量维度
    '''
    # embedd_dim = -1
    # embedd_dict = dict()
    # with open(embedding_path, 'r') as file:
    #     for line in file:
    #         line = line.strip()
    #         if len(line) == 0:
    #             continue
    #         tokens = line.split()
    #         if embedd_dim < 0:
    #             embedd_dim = len(tokens) - 1
    #         else:
    #             assert (embedd_dim + 1 == len(tokens))
    #         embedd = np.empty([1, embedd_dim])
    #         embedd[:] = tokens[1:]
    #         if sys.version_info[0] < 3:
    #             first_col = tokens[0].decode('utf-8')
    #         else:
    #             first_col = tokens[0]
    #         embedd_dict[first_col] = embedd
    # return embedd_dict, embedd_dim

def str2bool(string):
    """
    字符串与bool类型数据相互转换
    将True，true和TRUE转换为bool类的True
    其余字符串转换为bool类的False
    """     
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False

def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
    将输入的instance进行划分
    input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
        words: word ids for one sentence. (batch_size, sent_len)
        chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
    output:
        zero padding for word and char, with their batch length
        word_seq_tensor: (batch_size, max_sent_len) Variable
        word_seq_lengths: (batch_size,1) Tensor
        char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
        char_seq_lengths: (batch_size*max_sent_len,1) Tensor
        char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
        label_seq_tensor: (batch_size, max_sent_len)
        mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    with torch.no_grad():
        word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long())
    with torch.no_grad():
        mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    with torch.no_grad():
        char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len))).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label
    
def predict_check(pred_variable, gold_variable, mask_variable):
    """
        比较预测结果和真值ground truth,返回正确预测的个数
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
        output:  
            right_token: 正确预测的个数
            total_token: 总的样本个数  

    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask).astype('uint64')
    total_token = mask.sum().astype('uint64')
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token
