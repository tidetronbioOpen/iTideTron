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
import requests
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
    for row in gene_name:
        line = row.split(',')
        if '_' in line[0]:
            b= line[0].split('_')[0:2]
            a=b[0]+'_'+b[1]
        else:
            a = line[0]
        name.append(a)
    return name

def get_gene_info(name:str):
    """_summary_

    Args:
        name (str): gene name

    Returns:
        _type_: _description_
    """
    # 等待页面加载完成
    # 配置 Chrome WebDriver
    # s = Service('path/to/chromedriver')  # 替换为 chromedriver 可执行文件的路径
    s = Service("/home/GongFu/chuqi/chromedriver-linux64/chromedriver")
    options = Options()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('--headless')
    options.add_argument('--enable-javascript')  # 无头模式，可选择是否显示浏览器界面
    driver = webdriver.Chrome(service=s, options=options)
    url="https://www.ncbi.nlm.nih.gov/nuccore/"+name
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    target_divs = soup.find('div',id='viewercontent1', class_='seq gbff')
    target_divs_id = target_divs.attrs['val']
    info_url = 'https://www.ncbi.nlm.nih.gov/portal/loader/pload.cgi?curl=http%3A%2F%2Fwww.ncbi.nlm.nih.gov%2Fsviewer%2Fviewer.cgi%3Fid%3D'+ target_divs_id+\
            '%26db%3Dnuccore%26report%3Dgenbank%26conwithfeat%3Don%26hide-sequence%3Don%26hide-cdd%3Don%26retmode%3Dhtml%26ncbi_phid%3DCE8A379853270BA100000000011600A6%'+\
            '26withmarkup%3Don%26tool%3Dportal%26log%24%3Dseqview&pid=0'
    res = requests.get(info_url)
    driver.get(info_url)
    html = driver.page_source
    soup = BeautifulSoup(res.text, "html.parser")
    spans = soup.find_all('span')
    feature_id_list = []
    item_list = []
    for span in spans:
        feature_id = span.get('id').split('.')[1]
        feature_text = span.text
        data = {}
        for line in feature_text.split('\n')[1:]:
            if line.strip() and '=' in line:
                key, value = line.split('=', 1)
                data[key.strip()] = value.strip('"')

        feature_json = json.dumps(data, indent=2)
        item = (feature_id,feature_json)
        
        feature_id_list.append(feature_id)
        item_list.append(item)
    driver.quit()
    s.stop()
    logging.info(f"URL gene feature length: {len(feature_id_list)}")
    return item_list
    
    
def read_infor(gene_name_raw):
    """_summary_
        gene_name_raw:本地数据每条基因的名字,列表储存
        gene_names:爬取数据成功的基因名称，列表存储
        total_number:尝试爬取理数据条数,列表存储
        no_infor_record_list:爬取失败的基因,列表存储
        total_feature_text:获取每条基因所有特征信息,列表存储
    Returns:
        _type_: _description_
    """
    total_number = 0
    no_infor_record_list=[]#爬取失败数据记录
    total_feature_text = []
    gene_names = []
    for gene_item in gene_name_raw:
        try:
            item_list = get_gene_info(gene_item)#获取基因信息
        except AttributeError:
            logging.info(gene_item,'基因爬取过程出现问题。')
            no_infor_record_list.append(gene_item)
        else:
            total_number+=1
            total_feature_text.append(item_list)
            gene_names.append(gene_item)
        
    return total_feature_text,gene_name,no_infor_record_list

if __name__=='__main__':
    
    #读取外部传参
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',  help='Configuration File' )
    args = parser.parse_args()
    with open(args.config,encoding='utf-8') as f:#将yaml文件写为字典形式
        config = yaml.safe_load(f)

    #读取本地数据
    DIR_YES = config['raw_data_dir']
    gene_name = read(DIR_YES)

    #获取基因信息
    no_infor_list = []
    total_feature_text,gene_names,no_infor_list=\
        read_infor(gene_name)#基因信息获取
        
    dicts = [dict(inner_list) for inner_list in total_feature_text]
    data = {}
    for d in dicts:
        data.update(d)
    df = pd.DataFrame(data, index=gene_names)
    df.to_csv('scripts/result_predict.csv')
    # NO_INFOR_NUM=len(no_infor_list)#记录第一遍爬取过程中的未爬取信息基因条数
    
    #输出检测结果
    logging.info('总爬取基因数量:',len(gene_names))
    logging.info('未爬取成功的基因:',len(no_infor_list))
