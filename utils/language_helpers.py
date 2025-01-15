"""
Filename: language_helpers.py
Author: Wang Chuyu
Contact: wangchuyu@kongfoo.cn
"""
import collections
# import re
import numpy as np

def tokenize_string(sample):
    """
    将字符串分词为一个由小写单词组成的元组。

    参数：
        sample(str)：输入的字符串。

    返回：
        tuple：由小写单词组成的元组。

    示例：
        >>> tokenize_string("Hello, World!")
        ('hello,', 'world!')
    """
    return tuple(sample.lower().split(' '))

def mutation_callback(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.choice(["1", "2", "3", "4"])
    return chromosome

def num_to_seq(num_seq):
  atgc = {1:'A', 2:'T', 3:'G', 4:'C'}
  gene_seq = []
  for n in num_seq:
    gene_seq.append(atgc[n])
  return ''.join(gene_seq)

def seq_to_num(seq):
    num_seq = []
    atgc = {'A': 1, 'T': 2, 'G': 3, 'C': 4}
    for s in seq:
        num_seq.append(atgc[s])
    return num_seq


def is_valid_codon_seq(seq):
    dna_stop_codons = ['TAA', 'TAG', 'TGA']
    dna_codon_table = [
    'TTT', 'TTC', 'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG', 
    'ATT', 'ATC', 'ATA', 'ATG', 'GTT', 'GTC', 'GTA', 'GTG',
    'TCT', 'TCC', 'TCA', 'TCG', 'CCT', 'CCC', 'CCA', 'CCG',
    'ACT', 'ACC', 'ACA', 'ACG', 'GCT', 'GCC', 'GCA', 'GCG',
    'TAT', 'TAC', 'TAA', 'TAG', 'CAT', 'CAC', 'CAA', 'CAG',
    'AAT', 'AAC', 'AAA', 'AAG', 'GAT', 'GAC', 'GAA', 'GAG',
    'TGT', 'TGC', 'TGA', 'TGG', 'CGT', 'CGC', 'CGA', 'CGG',
    'AGT', 'AGC', 'AGA', 'AGG', 'GGT', 'GGC', 'GGA', 'GGG'
    ]
    # 长度判断、分割等逻辑

    for i in range(0, len(seq), 3):
       codon = seq[i:i+3]
       
       if codon not in dna_codon_table:
           return False
           
       if codon in dna_stop_codons:
           if i != len(seq) - 3:
               return False
  # 添加判断最后一个密码子    
    last_codon = seq[-3:]
    if last_codon not in dna_stop_codons:
      return False           
    return True


class NgramLM(object):
    """
    实现一个N元语言模型，用于计算文本数据中的N元语法相关的统计信息和相似度度量，
    以及一个加载数据集的函数，用于从文件中读取数据并进行预处理。
    """
    def __init__(self, _n, samples, tokenize=False):
        """
        初始化N元语言模型。

        参数：
            _n(int)：N元的大小。
            samples(list)：包含多个字符串样本的列表。
            tokenize(bool)：是否进行分词处理，默认为False。
        """
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = _n
        self._kmers = self.k_mer('', _n)
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        """
        生成样本中的N元组。

        返回：
            generator：生成N元组的迭代器。
        """
        _n = self._n
        for sample in self._samples:
            for i in range(len(sample) - _n + 1):
                yield sample[i:i + _n]

    def unique_ngrams(self):
        """
        获取样本中的唯一N元组。

        返回：
            set：样本中唯一N元组的集合。
        """
        return set(self._ngram_counts.keys())
    
    def kmers(self):
        """
        生成所有满足_n的k-mer子序列模式。

        返回：
            所有满足_n的k-mer子序列模式。
        """
        return self._kmers

    def log_likelihood(self, ngram):
        """
        计算给定N元组的对数似然。

        参数：
            ngram(tuple)：需计算对数似然的N元组。

        返回：
            float：对数似然值。
        """
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, _p):
        """
        计算当前模型与另一模型之间的KL散度。

        参数：
            _p(NgramLanguageModel)：另一N元语言模型。

        返回：
            float：KL散度值。
        """
        log_likelihood_ratios = []
        for ngram in _p.ngrams():
            log_likelihood_ratios.append(_p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, _p):
        """
        计算当前模型与另一模型之间的余弦相似度。

        参数：
            _p(NgramLanguageModel)：另一N元语言模型。

        返回：
            float：余弦相似度值。
        """
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in _p.unique_ngrams():
            p_i = np.exp(_p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i ** 2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i ** 2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, _p):
        """
        计算当前模型相对于另一模型的精确度。

        参数：
            _p(NgramLanguageModel)：另一N元语言模型。

        返回：
            float：精确度值。
        """
        num = 0.
        denom = 0
        p_ngrams = _p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, _p):
        """
        计算当前模型相对于另一模型的召回率。

        参数：
            _p(NgramLanguageModel)：另一N元语言模型。

        返回：
            float：召回率值。
        """
        return _p.precision_wrt(self)

    def js_with(self, _p):
        """
        ""
        计算当前模型与另一模型之间的JS散度。

        参数：
            _p(NgramLanguageModel)：另一N元语言模型。

        返回：
            float：JS散度值。
        """
        log_p = np.array([_p.log_likelihood(ngram) for ngram in _p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in _p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([_p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5 * (kl_p_m + kl_q_m) / np.log(2)
    
    def k_mer(self, root, num):
        """
        使用深度优先搜索生成DNA序列的所有k-mer子序列。

        参数:
            root(str): 当前生成的子序列。
            num(int): 生成的k-mer的长度。

        返回:
            list: 所有生成的k-mer子序列。

        示例:
            k_mer('', 2) # 输出: ['AA', 'AC', 'AT', 'AG', 'CA', 'CC', 'CT', 'CG', 'TA', 'TC']
        """
        if num == 1:
            return [root + 'A', root + 'C', root + 'T', root + 'G']
        buf = [self.k_mer(root + 'A', num - 1), self.k_mer(root + 'C', num - 1),
            self.k_mer(root + 'T', num - 1), self.k_mer(root + 'G', num - 1)]
        res = []
        for _it in buf:
            res.extend(_it)
        return res
    
    def is_kmer(self):
        return set(self._samples).issubset(set(self._kmers))


def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048, data_dir=''):
    """
    从文件中加载数据集，并进行预处理，返回处理后的数据集、字符映射表和反向字符映射表。

    参数：
        max_length(int)：最大样本长度。
        max_n_examples(int)：加载的最大样本数量。
        tokenize(bool)：是否进行分词处理，默认为False。
        max_vocab_size(int)：字符映射表的最大大小，默认为2048。
        data_dir(str)：数据文件所在的目录，默认为空字符串。

    返回：
        tuple：包含处理后的数据集、字符映射表和反向字符映射表的元组。

    示例：
        >>> load_dataset(100, 1000)
    """
    print("loading dataset...")

    lines = []

    finished = False

    path = data_dir + "/cRBS_long.txt"
    with open(path, 'r', encoding='utf-8') as _f:
        for line in _f:
            line = line[:-1]
            if tokenize:
                line = tokenize_string(line)
            else:
                line = tuple(line)

            if len(line) > max_length:
                line = line[:max_length]

            lines.append(line + (("`",) * (max_length - len(line))))

            if len(lines) == max_n_examples:
                finished = True
                break

    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk': 0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(max_vocab_size - 1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in range(100):
        print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap
