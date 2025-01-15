"""
Filename: calculator.py
Author: Li Yi
Contact: liyi@kongfoo.cn
"""
from collections import Counter,defaultdict
import math
class CAICalculator:
    """
    用于计算密码子优化序列与基准序列的CAI(Codon Adaptation Index)。
    """

    def __init__(self, reference_genes, rscu=None):
        """
        初始化CAI_calculator对象。

        参数:
            reference_genes(str): 基准序列，例如基准基因组或转录组序列。
            new_gene(str): 待测序列。
            rscu(dict): 密码子权重字典，可选参数。键是密码子，值是对应的权重。
        """
        self.reference_genes = reference_genes
        self.rscu = rscu if rscu else {}
        self.calculate_RSCU()

    def calculate_RSCU(self):
        """
        计算参考基因组中每种密码子在同义密码子中的使用频率
        返回一个字典，key为氨基酸密码子对组合，如M j
        value为该氨基酸密码子对组合对应的密码子在同义密码子中的使用频率
        """
        codon_count = defaultdict(int)
        
        # Count codons
        for gene in self.reference_genes:
            lines=gene.split('\n')
            for line in lines:
                if line=='' or line[0] == '#':
                    continue
                amino_acid=line[0]
                codon_box=line[2]
                codon_count[line] += 1
        # Get synonymous codon counts
        syn_counts = defaultdict(int)
        for line,count in codon_count.items():
            amino_acid=line[0] # first letter is amino acid
            syn_counts[amino_acid] += count

        # Calculate RSCU
        for line,count in codon_count.items():
            amino_acid=line[0]
            self.rscu[line] = count / syn_counts[amino_acid]


    def calculate_CAI(self,new_gene):
        """
        输入参数：
        new_gene:需要计算CAI的序列，格式为包含一个或多个字符串的list
        输出：
        cais:包含每个序列相对参考基因组的CAI分数的list
        """
        # Calculate CAI
        cais=[]
        for gene in new_gene:
            cai_vals = []

            gene=gene.splitlines() 
            for codon in gene:
                if codon=='' or codon=='\n' or codon[0]=='#':
                    continue
                assert len(codon)==3
                RSCU = self.rscu[codon[0:3]] 
                max_rscu = max(self.rscu[gene] for gene in self.rscu if gene.startswith(codon[0]))
                rel_adapt = RSCU/max_rscu
                cai_vals.append(rel_adapt)
            cai = math.exp(sum(math.log(v) for v in cai_vals)/len(cai_vals))
            cais.append(cai)
        return cais
