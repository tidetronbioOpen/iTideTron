"""
Filename: dataset.py
Author: Li Yi
Contact: liyi@kongfoo.cn
"""
class Dataset:
    """
    Dataset基类,定义了init,load,read_config以及save方法
    """
    def __init__(self,config_file):
        """
        Dataset类实例初始化
        """
        super(Dataset, self).__init__()
        pass

    def load(self,data_file=None):
        """
        从data_file路径加载数据创建数据集
        """
        pass

    def save(self,save_file=None):
        """
        将数据保存在save_file路径下
        """        
        pass

    def read_config(self,config):
        """
        从config中读取配置
        """
        pass
