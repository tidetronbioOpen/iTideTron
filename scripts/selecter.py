
"""
Filename: selecter.py
Author: Li Yi
Contact: liyi@kongfoo.cn
"""
import pandas as pd
import numpy as np
import csv
import os
import argparse
import yaml


if __name__ == '__main__':
    """
    Preprocess the Ecoli codon data
    
    Usage: python selecter.py --config <config_file_path>
    
    Arguments:
        --config (str): Path to the configuration file containing model settings.

    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', help='Configuration File', default='config/codon_train.yml' )
    args = parser.parse_args()
    config = args.config
    with open(config) as f:#将yaml文件写为字典形式
        config = yaml.safe_load(f)
    current_path = os.path.dirname(os.path.abspath(__file__))
    root_dir=os.path.join(current_path,'..')
    data_path=os.path.join(root_dir,config['data_path'])
    assert os.path.exists(data_path)==True
    df = pd.read_excel(data_path)
    df = df.drop_duplicates(subset='gene_id', keep='first')
    df = df.sort_values(by='T0_T1_prot/rna_log10')
    df = df.reset_index(drop=True)
    threshold1 = df['T0_T1_prot/rna_log10'].quantile(q=config['high_pre'],interpolation='linear')
    threshold2 = df['T0_T1_prot/rna_log10'].quantile(q=config['mid_pre'],interpolation='linear')
    threshold3 = df['T0_T1_prot/rna_log10'].quantile(q=config['low_pre'],interpolation='linear')
    selected1 = df[df['T0_T1_prot/rna_log10'] > threshold1][['gene_id', 'cds_seqs']]
    selected2 = df[(threshold3 < df['T0_T1_prot/rna_log10']) & (df['T0_T1_prot/rna_log10'] <= threshold2)][['gene_id', 'cds_seqs']]
    selected3 = df[df['T0_T1_prot/rna_log10'] <=threshold3][['gene_id', 'cds_seqs']]
    selected = pd.concat([selected1,selected3])
    selected = selected.rename(columns={'gene_id':'names','cds_seqs':'feature'})
    selected_path=os.path.join(root_dir,config['raw_dir'])
    selected_data_path=os.path.join(root_dir,config['selected_data_path'])
    initial_population_path=os.path.join(root_dir,config['initial_population_path'])
    selected['feature'].to_csv(initial_population_path,header=False,index=False)
    selected.to_csv(selected_path,index=False)
    assert os.path.exists(selected_path)==True
    with open(selected_path) as in_file:
        with open(selected_data_path, 'w', newline='') as out_file:
            writer = csv.writer(out_file, quotechar='"', quoting=csv.QUOTE_ALL)
            for line in in_file:
                writer.writerow([line.strip()])
    assert os.path.exists(selected_data_path)==True
    f=open(selected_data_path,'r')
    lines=f.readlines()
    f.close()
    f=open(selected_path,'w')
    f.write('line\n')
    f.writelines(lines)
    f.close()
    selected1_RNA_score = df[df['T0_T1_prot/rna_log10'] > threshold1][['gene_id', 'rna_struct_score']]
    selected2_RNA_score = df[(threshold3 < df['T0_T1_prot/rna_log10']) & (df['T0_T1_prot/rna_log10'] <= threshold2)][['gene_id', 'rna_struct_score']]
    selected3_RNA_score = df[df['T0_T1_prot/rna_log10'] <=threshold3][['gene_id', 'rna_struct_score']]
    selected_RNA_score = pd.concat([selected1_RNA_score,selected3_RNA_score]) 
    gene_type=config['type']
    if gene_type.lower()=='codon':
        rna_path=os.path.join(root_dir,config['rna_score'])
        selected_RNA_score.to_csv(rna_path, index=False)
        assert os.path.exists(rna_path)==True
    names = pd.concat([selected1, selected3])['gene_id']
    tags = [1] * len(selected1) + [0] * len(selected3)
    tag_path=os.path.join(root_dir,config['labels_dir'])
    f = open(tag_path, 'w')
    writer = csv.writer(f)
    writer.writerow(names)
    writer.writerow(tags)
    f.close()
    assert os.path.exists(tag_path)==True