"""
Filename: alphabet.py
Author: Li Yi
Contact: liyi@kongfoo.cn
"""


from __future__ import print_function
import json
import os
import sys


class Alphabet:
    """
    字母表类，用于将字符序列转化为数字序列
    内容主要是元素索引表，每个元素instance有对应index索引
    Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
    """
    def __init__(self, name, label=False, keep_growing=True):
        """
        字母表类初始化
        初始化时可以传入的参数有
        1.name字母表名，包括word，char，label等
        2.keep_growing增长状态，默认为True
        除label alphabet(密码子对字母表)外其他字母表初始化时的默认元素为unk，代表unknown
        label参数代表是否是label alphabet(密码子对字母表)
        """
        self.name = name
        self.UNKNOWN = "</unk>"
        self.label = label
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1
        if not self.label:
            self.add(self.UNKNOWN)

    def clear(self, keep_growing=True):
        """
        清除字母表的内容，维持字母表的增长状态
        """
        # self.instance2index = {}
        # self.instances = []
        # self.keep_growing = keep_growing
        # # Index 0 is occupied by default, all else following.
        # self.default_index = 0
        # self.next_index = 1

    def add(self, instance):
        """
        添加一个元素到字母表中
        如果输入的元素不在字母表内，则字母表新增一条元素索引对应关系
        否则字母表不变化
        """
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        """
        输入一个元素，返回该元素在字母表中的索引
        如果该元素在字母表中，则字母表不变化
        如果该元素不在字母表中，且字母表处于增长状态，则新增一条元素索引对应关系
        如果该元素不在字母表中，且字母表不处于增长状态，则返回unk的索引
        """
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        """
        输入一个索引，返回索引对应的元素
        """
        if index == 0:
            # First index is occupied by the wildcard element.
            return None
        try:
            return self.instances[index - 1]
        except IndexError:
            print('WARNING:Alphabet get_instance ,unknown instance, return the first label.')
            return self.instances[0]

    def size(self):
        """
        返回字母表大小
        计算方法为元素个数(包括unk)+1
        需要加1是因为unk的index为1，整个字母表index从1开始增长
        程序运行过程中数据有补0的操作
        """
        # if self.label:
        #     return len(self.instances)
        # else:
        return len(self.instances)+1

    def iteritems(self):
        """
        返回元素索引表内容的列表形式
        """
        return self.instance2index.items()

    # def enumerate_items(self, start=1):
        # if start < 1 or start >= self.size():
        #     raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        # return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        """
        关闭字母表，字母表不再添加元素，增长状态变为False
        """
        self.keep_growing = False

    def open(self):
        """
        开启字母表，字母表开始添加元素，增长状态变为False
        """
        self.keep_growing = True

    def get_content(self):
        """
        返回字典形式的字母表内容
        字典中instance2index为字母表的元素索引对
        instances为所有的元素
        """
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        """
        从json文件读取字母表
        """
        # self.instances = data["instances"]
        # self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        # saving_name = name if name else self.__name
        # try:
        #     json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        # except Exception as e:
        #     print("Exception: Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        # loading_name = name if name else self.__name
        # self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
