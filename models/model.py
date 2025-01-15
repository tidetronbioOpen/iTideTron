"""
Filename: model.py
Author: Li Yi
Contact: liyi@kongfoo.cn
"""

class Model:
    """
    Model基类,定义了train,以及predict方法
    """
    def __init__(self,config_file):
        """
        Modle类实例初始化
        """
        super(Model, self).__init__()
        pass

    def train(self,data_file=None):
        pass

    def predict(self,save_file=None):
        pass