"""
Filename: download.py
Author: KONGFOO 
Contact: dev@kongfoo.cn
"""
import os, logging
import concurrent.futures
from tqdm import tqdm

class GeneDownloader:
    """
    用于下载基因序列数据的类。
    """

    def __init__(self, gene_file_dir, save_file_name, ncores=1):
        """
        初始化GeneDownloader类的实例。

        参数：
        - gene_file_dir(str): 保存待下载基因序列的文件名。
        - save_file_name(str): 保存下载得到基因序列的文件名。
        """
        # gene_ids(list): 待下载的所有基因ID。
        self._gene_file_dir = gene_file_dir
        self._file_name = save_file_name 

        with open(self._file_name, mode='w') as file:
            file.write('line\n')
            file.write('"name,feature"\n')

        gene_file_list = []
        for f in tqdm(sorted(os.listdir(gene_file_dir))):
            if f.endswith('.fna') and os.path.isfile(os.path.join(gene_file_dir, f)):
                #gene_file_list.append(f)
                self._from_fna_file(f)

        # with concurrent.futures.ThreadPoolExecutor(max_workers=ncores) as executor:
        #     # Read lines in parallel and process them
        #     futures = [executor.submit(self._from_fna_file, f) for f in tqdm(gene_file_list)]
        #     # Wait for all tasks to complete
        #     concurrent.futures.wait(futures)    

    def _from_fna_file(self, fna_file_name):
        """
        将.fna格式的追加到文件新的一行 

        返回：
        """
        #file_name = os.path.basename(fna_file_path)
        gene_name = '_'.join(fna_file_name.split('_')[0:-1])
        logging.info(f'Start processing {gene_name}...')
        gene_seq = ''

        with open(os.path.join(self._gene_file_dir, fna_file_name), mode='r',
                  encoding='utf-8', errors='replace') as file:
            # Read the first line
            line = file.readline()
            # Read and process the remaining lines
            for line in file:
                logging.debug(line)
                gene_seq += line.rstrip('\n')

        logging.debug(gene_seq)
        # Open the file in write mode
        with open(self._file_name, mode='a') as file:
            file.write('"'+gene_name+','+gene_seq+'"\n')

        logging.info(f'End processing {gene_name}...')
        