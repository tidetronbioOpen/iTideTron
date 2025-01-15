import time
import numpy as np
from utils.functions import recover_nbest_label,recover_label,batchify_with_label
from nnet.seqmodel import get_ranking_loss_binary_classification,expression_pred
def reverse_style(input_string):
    '''
    将字符串中的样式反转
    :param input_string: 待反转的字符串
    :return: 反转后的字符串
    '''
    # target_position = input_string.index('[')
    # input_len = len(input_string)
    # output_string = input_string[target_position:input_len] + input_string[0:target_position]
    # return output_string


# 输入参数 label_list:标签列表
def get_ner_BMES(label_list):
    '''
    此函数用于从序列标注的标签列表中提取命名实体识别的BMES标签
    输入参数label_list为标签列表
    输出结果stand_matrix为处理后的标签列表

    算法流程:
    1. 定义BMES标签类型
    2. 遍历标签列表,根据标签类型构建实体标签
    3. 将未结束的实体标签在最后补全
    4. 对标签进行后处理,生成结果stand_matrix

    BMES标签表示:
    B - 命名实体开始词
    M - 命名实体中间词
    E - 命名实体结束词  
    S - 单独成词的命名实体
    '''
    
    # # 初始化变量
    # list_len = len(label_list) 
    # begin_label = 'B-'
    # end_label = 'E-'
    # single_label = 'S-'
    # whole_tag = ''
    # index_tag = ''
    # tag_list = []
    # stand_matrix = []

    # # 遍历标签列表
    # for i in range(0, list_len):
    #     current_label = label_list[i].upper()
        
    #     # 判断当前标签类型
    #     if begin_label in current_label:
    #         # 如果前面已经有未结束的实体,先添加到列表
    #         if index_tag!= '':
    #             tag_list.append(whole_tag + ',' + str(i-1))
            
    #         # 更新whole_tag和index_tag
    #         whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
    #         index_tag = current_label.replace(begin_label,"",1)

    #     elif single_label in current_label:
    #         # 单独成词实体处理
    #         if index_tag!= '':
    #             tag_list.append(whole_tag + ',' + str(i-1))
    #         whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
    #         tag_list.append(whole_tag)
    #         whole_tag = ""
    #         index_tag = ""
        
    #     elif end_label in current_label:
    #         # 结束词处理
    #         if index_tag!= '':
    #             tag_list.append(whole_tag +',' + str(i))
    #         whole_tag = ''
    #         index_tag = ''
        
    #     else:
    #         # 非实体词
    #         continue

    # # 如果标签尚未结束,添加最后一个实体
    # if (whole_tag!= '')&(index_tag!= ''):
    #     tag_list.append(whole_tag)

    # # 后处理标签列表,生成stand_matrix
    # for i in range(0, len(tag_list)):
    #     if len(tag_list[i]) > 0:
    #         tag_list[i] = tag_list[i]+ ']'
    #         insert_list = reverse_style(tag_list[i])
    #         stand_matrix.append(insert_list)

    # return stand_matrix

def get_ner_BIO(label_list):
    '''
    此函数用于从序列标注的标签列表中提取命名实体识别的BIO标签
    输入参数label_list为标签列表 
    输出结果stand_matrix为处理后的标签列表

    算法流程:  
    1. 定义BIO标签类型
    2. 遍历标签列表,根据标签构建实体标签
    3. 将未结束的实体标签在最后补全 
    4. 对标签进行后处理,生成结果stand_matrix

    BIO标签表示:
    B - 命名实体开始词
    I - 命名实体中间词
    O - 非实体词
    
    '''    
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        # if begin_label in current_label:
        #     if index_tag == '':
        #         whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
        #         index_tag = current_label.replace(begin_label,"",1)
        #     else:
        #         tag_list.append(whole_tag + ',' + str(i-1))
        #         whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
        #         index_tag = current_label.replace(begin_label,"",1)

    #     elif inside_label in current_label:
    #         if current_label.replace(inside_label,"",1) == index_tag:
    #             whole_tag = whole_tag
    #         else:
    #             if (whole_tag != '')&(index_tag != ''):
    #                 tag_list.append(whole_tag +',' + str(i-1))
    #             whole_tag = ''
    #             index_tag = ''
    #     else:
    #         if (whole_tag != '')&(index_tag != ''):
    #             tag_list.append(whole_tag +',' + str(i-1))
    #         whole_tag = ''
    #         index_tag = ''

    # if (whole_tag != '')&(index_tag != ''):
    #     tag_list.append(whole_tag)
    # tag_list_len = len(tag_list)

    # for i in range(0, tag_list_len):
    #     if  len(tag_list[i]) > 0:
    #         tag_list[i] = tag_list[i]+ ']'
    #         insert_list = reverse_style(tag_list[i])
    #         stand_matrix.append(insert_list)
    return stand_matrix



def readSentence(input_file):
    '''
    读取输入文件，并将其转换为句子和标签
    :param input_file: 输入文件
    :return: 句子列表，标签列表
    '''
    # in_lines = open(input_file,'r').readlines()
    # sentences = []
    # labels = []
    # sentence = []
    # label = []
    # for line in in_lines:
    #     if len(line) < 2:
    #         sentences.append(sentence)
    #         labels.append(label)
    #         sentence = []
    #         label = []
    #     else:
    #         pair = line.strip('\n').split(' ')
    #         sentence.append(pair[0])
    #         label.append(pair[-1])
    # return sentences,labels


def readTwoLabelSentence(input_file, pred_col=-1):
    '''读取输入文件，并将其转换为双标签格式'''
    # in_lines = open(input_file,'r').readlines()
    # sentences = []
    # predict_labels = []
    # golden_labels = []
    # sentence = []
    # predict_label = []
    # golden_label = []
    # for line in in_lines:
    #     if "##score##" in line:
    #         continue
    #     if len(line) < 2:
    #         sentences.append(sentence)
    #         golden_labels.append(golden_label)
    #         predict_labels.append(predict_label)
    #         sentence = []
    #         golden_label = []
    #         predict_label = []
    #     else:
    #         pair = line.strip('\n').split(' ')
    #         sentence.append(pair[0])
    #         golden_label.append(pair[1])
    #         predict_label.append(pair[pred_col])

    # return sentences,golden_labels,predict_labels


def fmeasure_from_file(golden_file, predict_file, label_type="BMES"):
    '''
    获取预测结果与真值的F-measure
    :param golden_file: ground truth真值文件路径
    :param predict_file: 预测结果文件路径
    :param label_type: 标签格式，BMES/BIOES
    :return: P,R,F
    '''
    # print("Get f measure from file:", golden_file, predict_file)
    # print("Label format:",label_type)
    # golden_sent,golden_labels = readSentence(golden_file)
    # predict_sent,predict_labels = readSentence(predict_file)
    # P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    # print ("P:%sm R:%s, F:%s"%(P,R,F))



def fmeasure_from_singlefile(twolabel_file, label_type="BMES", pred_col=-1):
    '''
    计算两个label文件的f-measure
    :param twolabel_file: 文件路径
    :param label_type: 标签格式，BMES/BIOES
    :param pred_col: 预测列
    :return: P,R,F
    '''
    # sent,golden_labels,predict_labels = readTwoLabelSentence(twolabel_file, pred_col)
    # P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    # print ("P:%s, R:%s, F:%s"%(P,R,F))

def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    '''
    函数功能：计算命名实体识别结果的性能指标
    输入参数:
    golden_lists: 真实的标签列表
    predict_lists: 预测的标签列表  
    label_type: 使用的标签表示方法,BMES或BIO
    输出指标:
    accuracy: 准确率 
    precision: 精确率
    recall: 召回率
    f_measure: F1值
    主要步骤:
    1. 根据label_type调用相应函数,转换标签表示
    2. 提取真实和预测的命名实体标签
    3. 计算真正例和预测例的数量
    4. 计算准确率、精确率、召回率和F1值
    '''
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    # print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_measure

def evaluate(data, model, name, classifier,nbest=None):
    '''
    函数主要功能，根据输入的name返回模型在对应数据集上的准确率、精确率、召回率等数据
    :param data:密码子数据集，codondataset类实例
    :param model:标注模型,seqmodel类实例
    :param name:需要评测的数据集，包括train(训练集)dev(验证集)test(测试集)以及raw(待预测)
    :param nbest:如果是进行标注时调用，可以传入nbest参数，代表标注时输出的序列个数
    :return:speed, acc(accuracy), p(precision), r(recall), f(f_measure), pred_results(预测结果), pred_scores(预测分数)
    '''
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    expr_acc = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        if nbest:
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # print("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
        if name != 'raw':
            batch_ori_seq=model.batch_boxtocodon(batch_word,batch_label,mask,batch_wordrecover)
            ori_expression_results=expression_pred(batch_ori_seq,classifier)
            ori_expr_acc=1-get_ranking_loss_binary_classification(ori_expression_results,model.expected_result)/batch_word.size(0)
            batch_seq=model.batch_boxtocodon(batch_word,tag_seq,mask,batch_wordrecover)
            expression_results=expression_pred(batch_seq,classifier)
            expr_acc += 1-get_ranking_loss_binary_classification(expression_results,model.expected_result)/batch_word.size(0)-ori_expr_acc
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    expr_acc /= total_batch 
    if nbest:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores,expr_acc
    return speed, acc, p, r, f, pred_results, pred_scores,expr_acc
