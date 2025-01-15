from selenium import webdriver
import argparse
import yaml
import pandas as pd
import requests
import multiprocessing
from itertools import islice
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import json

def get_table(driver):
    
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    job_id = soup.find('td', class_='col-md-9').text
    table = driver.find_element(By.ID, 'report_table')
    # 获取所有行
    table_rows = table.find_elements(By.TAG_NAME, "tr") 
    commert_base_url = "http://bioinfor.imu.edu.cn/iprobiotics/public/Process?jobid="+\
        job_id + "&prediction="
    data = requests.get(commert_base_url)
    obj = json.loads(data.text)[0]
    row_name = obj['row_name'] 
    probiotics_score = obj['y_pro']
    non_probiotics_score = obj['n_pro']
    # data = json.loads(response.text)
    return row_name, probiotics_score, non_probiotics_score

def process_row(row):
    """
    计算每个子序列在样本中出现的频率。

    参数:
        row: 菌株名称和碱基序列。
    返回:
        已处理子序列的频率。

    示例:
    >>> process_row(row)
    """
    
    seq = row.split(',')[-1]
    name = row.split(',')[0]
    return name, seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',  help='Configuration File' )
    args = parser.parse_args()
    with open(args.config,encoding='utf-8') as f:#将yaml文件写为字典形式
        config = yaml.safe_load(f)
    # 等待页面加载完成
    # 配置 Chrome WebDriver
    s = Service("/home/GongFu/chuqi/chromedriver-linux64/chromedriver")  # 替换为 chromedriver 可执行文件的路径
    options = Options()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('--headless')
    options.add_argument('--enable-javascript')  # 无头模式，可选择是否显示浏览器界面
    driver = webdriver.Chrome(service=s, options=options)

    url = 'http://bioinfor.imu.edu.cn/iprobiotics/public/' + config['server']
    driver.get(url)

    data_path = config['raw_data']
    raw_data = pd.read_csv(data_path)
    raw_data.columns = ['line']
    raw_data = raw_data[1:]['line']
    submit_btn = driver.find_element(By.XPATH,'//*[@id="submit "]')
    
    # 将raw_data分割成要并行处理的行。
    # rows = islice(raw_data.iterrows(), 1, None)
    search_windows = driver.current_window_handle
    query_id_list = []
    probiotics_score_list = []
    non_probiotics_score_list = []
    wrong_pro_list = []
    for row in raw_data:
        name,sequence = process_row(row)
        input_genomic = '>'+name + '\n' + sequence
        seq_input = driver.find_element(By.ID,"fasta")
        seq_input.clear()
        seq_input.send_keys(input_genomic)
        submit_btn.click()
        # 获得当前所有打开的窗口的句柄
        all_handles = driver.window_handles
        # 跳转到新的想要跳转的页面
        for handle in all_handles:
            if handle == search_windows:
                # 切换到新的页面
                continue
            else:
                try:
                    driver.switch_to.window(handle)
                    query_id, probiotics_score, non_probiotics_score = get_table(driver)
                    query_id_list.append(query_id)
                    probiotics_score_list.append(probiotics_score)
                    non_probiotics_score_list.append(non_probiotics_score)
                    # 关闭新打开的窗口 
                    if driver.title == 'Report':
                        driver.close()
                    # 切换回之前的窗口
                    driver.switch_to.window(all_handles[0]) 
                    break
                except: 
                    wrong_pro_list.append(row)
                    break
                    
    driver.quit()
    result = pd.DataFrame()
    result['gene_id'] = query_id_list
    result['probiotics'] = probiotics_score_list
    result['non-probiotics'] = non_probiotics_score_list
    result.to_csv(config['result_data'],index=False)
    if len(wrong_pro_list) !=0 :
        wrong_serch = pd.DataFrame()
        wrong_serch['item'] = wrong_pro_list
        result.to_csv(config['wrong_data'],index=False)
        # break

