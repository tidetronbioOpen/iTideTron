"""
库说明
json:读写错误信息文件
argparse:外部传参
selenium:联网进行爬虫
bs4:数据清洗
pandas:数据处理
tqdm:日志中进度可视化
yaml:用于加载yaml文件
"""
import json, os
import argparse
import pandas as pd
import yaml
import logging
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

def read(path):
    """_summary_

    Args:
        path (str): 本地存储数据位置

    Returns:
        _type_: _description_
    """
    #读取已经下载的数据，获取需要的信息
    data = pd.read_csv(path)

    gene_name = list(data.iloc[1:, 0])
    name = []
    gene_len = []
    for row in gene_name:
        line = row.split(',')
        b= line[0].split('_')[0:2]
        a=b[0]+'_'+b[1]
        name.append(a)
        gene_seq_len=len(line[1])
        gene_len.append(gene_seq_len)
    return name,gene_len

def get_gene_length(name:str):
    """_summary_

    Args:
        name (str): gene name

    Returns:
        _type_: _description_
    """
    # 等待页面加载完成
    # 配置 Chrome WebDriver
    s = Service('path/to/chromedriver')  # 替换为 chromedriver 可执行文件的路径
    options = Options()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('--headless')
    options.add_argument('--enable-javascript')  # 无头模式，可选择是否显示浏览器界面
    driver = webdriver.Chrome(service=s, options=options)
    url="https://www.ncbi.nlm.nih.gov/datasets/genome/"+name
    driver.get(url)
    # 等待页面加载完成
    # driver.implicitly_wait(50)  # 设置最长等待时间，单位为秒

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    div_element =soup.find('td',\
    class_='MuiTableCell-root MuiTableCell-body MuiTableCell-sizeMedium TableCell-cell \
        TableCell-cellRightAlign TableCell-cellNoWrap css-cvb69c').text
    infor=int(div_element.replace(',',''))
    driver.quit()
    s.stop()
    #logging.info(f"URL gene length: {len(gene)}")
    return infor

def check_length(gene_data:tuple,\
                number_tuple:tuple,\
                config_dict:dict,wrong_record_list:list,isfirst:bool):
    """_summary_
        gene_data:
            gene_len:本地数据每条基因长度，列表储存
            gene_name:本地数据每条基因名字，列表储存
        number_tuple:
            total_number:尝试爬取理数据条数
            right_number:核对正确数据条数
            wrong_number:核对错误数据条数
        config_dict:保存路径参数与数据路径参数
        wrong_record_list:核对结果为错误数据记录
        isfirst:是否第一遍爬取数据
    Returns:
        _type_: _description_
    """
    gene_len,gene_name=gene_data
    no_infor_record_list=[]#爬取失败数据记录
    total_number,right_number,wrong_number=number_tuple
    for gene_len_item,gene_name_item in tqdm(zip(gene_len,gene_name)):
        try:
            real_gene_length = get_gene_length(gene_name_item)#获取基因真实长度
        except AttributeError:
            logging.info(gene_name_item,'基因爬取过程出现问题。')
            no_infor_record_list.append([gene_name_item,gene_len_item])
        else:
            total_number+=1
            try:
                assert real_gene_length == gene_len_item
                logging.info('name:',gene_name_item,'len:',gene_len_item,'real_len:',real_gene_length)
            except AssertionError:
                wrong_number+=1
                wrong_record_list.append(gene_name_item)
                logging.info('name:',gene_name_item,'len:',gene_len_item,'real_len:',real_gene_length)
            else:
                right_number+=1
    #保存文件：
    if isfirst:
        with open(config_dict['wrong_gene_save_path'],'w',encoding='utf-8') as wrongfile:
            json.dump(wrong_list, wrongfile)
    else:
        with open(config_dict['wrong_gene_save_path'],'w+',encoding='utf-8') as wrongfile:
            json.dump(wrong_list, wrongfile)

    with open(config_dict['no_infor_gene_save_path'],'w',encoding='utf-8') as noinforfile:
        json.dump(no_infor_list, noinforfile)
    return total_number,right_number,wrong_number,no_infor_record_list,wrong_record_list

if __name__=='__main__':
    #读取外部传参
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',  help='Configuration File' )
    args = parser.parse_args()
    with open(args.config,encoding='utf-8') as f:#将yaml文件写为字典形式
        config = yaml.safe_load(f)

    #读取本地数据
    DIR_YES = config['raw_data_dir']
    gene_name_yes, gene_len_yes = read(DIR_YES)

    #判断下载基因长度与基因真实长度是否一致
    TOTAL_NUM=0#总基因数
    RIGHT_NUM=0#通过检验基因数目
    WRONG_NUM=0#未通过检验基因数目
    wrong_list=[]#未通过基因记录

    TOTAL_NUM,RIGHT_NUM,WRONG_NUM,no_infor_list,wrong_list=\
        check_length((gene_len_yes,gene_name_yes),(TOTAL_NUM,\
                     RIGHT_NUM,WRONG_NUM),config,wrong_list,True)#长度检测

    NO_INFOR_NUM=len(no_infor_list)#记录第一遍检测过程中的未爬取信息基因条数

    COUNT=0#循环退出器
    while COUNT<3:
        with open(config['no_infor_gene_save_path'],'r',encoding='utf-8') as f:
            data_list = json.load(f)
        gene_name_yes=[data[0] for data in data_list]
        gene_len_yes=[data[1] for data in data_list]


        TOTAL_NUM,RIGHT_NUM,WRONG_NUM,no_infor_list,wrong_list=\
            check_length((gene_len_yes,gene_name_yes),(TOTAL_NUM,\
                         RIGHT_NUM,WRONG_NUM),config,wrong_list,False)#长度检测
        if len(no_infor_list)==0:
            break
        if NO_INFOR_NUM==len(no_infor_list):#这次无信息数据量与上次一致
            COUNT+=1
        else:
            COUNT=0

    #输出检测结果
    logging.info('总基因数:',WRONG_NUM+RIGHT_NUM)
    logging.info('通过检验基因数:',RIGHT_NUM)
    logging.info('未通过检验基因数：',WRONG_NUM)
    logging.info(f'通过率：{RIGHT_NUM*100/(RIGHT_NUM+WRONG_NUM)}%')
