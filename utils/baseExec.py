#!/usr/bin/env python

import sys
import os
class BaseExec:
    svmscale_file_name = ''
    svmtrain_file_name = ''
    svmpredict_file_name = ''
    gnuplot_full_path = ''
    grid_py_file_name = ''
    exec_base_path = ''
    exec_base_path_default = '../'
    py_base_path = ''
    py_base_path_default = './'
    def __init__(self, exec_base_path,py_base_path):
        self.exec_base_path = exec_base_path
        self.py_base_path = py_base_path
        # svm, grid, and gnuplot executable files example:  '../svm-scale' './grid.py'
        is_win32 = (sys.platform == 'win32')
        self.grid_py_file_name = r'grid.py'
        if not is_win32:
            self.svmscale_file_name = 'svm-scale'
            self.svmtrain_file_name = 'svm-train'
            self.svmpredict_file_name = 'svm-predict'
            self.gnuplot_full_path = '/usr/bin/gnuplot'
            self.exec_base_path_default = '../'
            self.py_base_path_default = './'
        else:
            # example for windows example:  r'..\windows\svm-scale.exe' r'.\grid.py'
            self.svmscale_file_name = r'svm-scale.exe'
            self.svmtrain_file_name = r'svm-train.exe'
            self.svmpredict_file_name = r'svm-predict.exe'
            self.gnuplot_full_path = r'c:\tmp\gnuplot\binary\pgnuplot.exe'
            self.exec_base_path_default = r'..\windows\\'
            self.py_base_path_default = r".\\"

    def apply_base_path(self):
        if not os.path.exists(self.exec_base_path):
            self.exec_base_path = self.exec_base_path_default
        if not os.path.exists(self.py_base_path):
            self.py_base_path = self.py_base_path_default

        self.svmscale_file_name = os.path.join(self.exec_base_path, self.svmscale_file_name)
        self.svmtrain_file_name = os.path.join(self.exec_base_path, self.svmtrain_file_name)
        self.svmpredict_file_name = os.path.join(self.exec_base_path, self.svmpredict_file_name)
        self.grid_py_file_name = os.path.join(self.py_base_path, self.grid_py_file_name)

    def validate_paths(self):
        # if not os.path.exists(self.svmscale_path):print("svm-scale executable not found")
        # if not os.path.exists(self.svmtrain_path):print("svm-train executable not found")
        # if not os.path.exists(self.svmpredict_path):print("svm-predict executable not found")
        # if not os.path.exists(self.gnuplot_path):print("gnuplot executable not found")
        # if not os.path.exists(self.grid_py_path):print("grid.py not found")
        print("base validation")
        assert os.path.exists(self.svmscale_file_name),"svm-scale executable not found"
        assert os.path.exists(self.svmtrain_file_name),"svm-train executable not found"
        assert os.path.exists(self.svmpredict_file_name),"svm-predict executable not found"
        assert os.path.exists(self.gnuplot_full_path),"gnuplot executable not found"
        assert os.path.exists(self.grid_py_file_name),"grid.py not found"
        