"""
Filename: mutation.py
Author: Wang Chuyu
Contact: wangchuyu@kongfoo.cn
"""
import random

class MutationGenerator:
    """
    用于生成随机突变序列的类。
    """

    def __init__(self, sequence, mutation_probabilities):
        """
        初始化mutation_generator类的实例。

        参数：
        - sequence(str): 输入的碱基序列。
        - mutation_probabilities(dict): 突变碱基对的概率字典。
        """
        self.sequence = sequence
        self.mutation_probabilities = mutation_probabilities

    def generate_mutation(self):
        """
        生成随机突变序列。

        返回：
        - mutation_sequence(str): 生成的突变序列。
        """
        mutation_sequence = ""
        for seq in self.sequence:
            mutation_sequence += self.mutate(seq)
        return mutation_sequence

    def mutate(self, base):
        """
        随机突变一个碱基。

        参数：
        - base(str): 当前碱基。

        返回：
        - mutation(str): 突变后的碱基。
        """
        mutation = base
        mutation_prob = self.mutation_probabilities.get(base, {})
        total_prob = sum(mutation_prob.values())
        random_prob = random.uniform(0, total_prob)
        cumulative_prob = 0

        for seq, prob in mutation_prob.items():
            cumulative_prob += prob
            if random_prob <= cumulative_prob:
                mutation = seq
                break

        return mutation

# 调用示例
# SEQ = "ATGCATGCATGCATGC"
# mutation_p = {
#     "A": {"A": 0.25, "T": 0.25, "G": 0.25 ,"C": 0.25},
#     "T": {"A": 0.25, "T": 0.25, "G": 0.25 ,"C": 0.25},
#     "G": {"A": 0.25, "T": 0.25, "G": 0.25 ,"C": 0.25},
#     "C": {"A": 0.25, "T": 0.25, "G": 0.25 ,"C": 0.25}
# }

# generator = MutationGenerator(SEQ, mutation_p)
# mut = generator.generate_mutation()
# print("原始序列:", SEQ)
# print("突变序列:", mut)
