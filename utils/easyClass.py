#!/usr/bin/env python

import subprocess
import sys
import os
import logging
from utils.baseExec import BaseExec

class Easy(BaseExec):
    # train_pathname = sys.argv[1]
    train_path = ''
    # test_pathname = sys.argv[2]
    test_path = ''
    cmd = ''
    scaled_file = ''
    model_file = ''
    range_file = ''

    scaled_test_file = ''
    predict_test_file = ''

    def __init__(self, exec_base_path,py_base_path,train_path,test_path):
        super().__init__(exec_base_path,py_base_path)
        super().apply_base_path()
        self.train_path = train_path
        self.test_path = test_path
        
    def validate_paths(self):
        super().validate_paths()
        logging.info("easy validation")
        assert os.path.exists(self.train_path),"training file not found"
        if not self.test_path == '':
            assert os.path.exists(self.test_path),"testing file not found"

    def create_train_files_path(self):
        #assert os.path.exists(self.train_path),"training file not found"
        file_name = os.path.split(self.train_path)[1]
        self.scaled_file = file_name + ".scale"
        self.model_file = file_name + ".model"
        self.range_file = file_name + ".range"

# # train_pathname = sys.argv[1]
# # assert os.path.exists(train_pathname),"training file not found"
# # file_name = os.path.split(train_pathname)[1]
# # scaled_file = file_name + ".scale"
# # model_file = file_name + ".model"
# # range_file = file_name + ".range"

    def create_test_files_path(self):
        if not self.test_path == '':
            file_name = os.path.split(self.test_path)[1]
            #assert os.path.exists(self.test_path),"testing file not found"
            self.scaled_test_file = file_name + ".scale"
            self.predict_test_file = file_name + ".predict"

# # if len(sys.argv) > 2:
# #     test_pathname = sys.argv[2]
# #     file_name = os.path.split(test_pathname)[1]
# #     assert os.path.exists(test_pathname),"testing file not found"
# #     scaled_test_file = file_name + ".scale"
# #     predict_test_file = file_name + ".predict"

    def perform_svmscale_scaling_training_data(self):
        logging.info("perform_svmscale_scaling_training_data")
        self.cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(self.svmscale_file_name, self.range_file, self.train_path, self.scaled_file)
        logging.info('Scaling training data...')
        subprocess.Popen(self.cmd, shell = True, stdout = subprocess.PIPE).communicate()


# # cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, train_pathname, scaled_file)
# # logging.info('Scaling training data...')
# # Popen(cmd, shell = True, stdout = PIPE).communicate()

    def perform_grid_py_cross_validation(self):
        self.cmd = '{0} -svmtrain "{1}" -gnuplot "{2}" "{3}"'.format(self.grid_py_file_name, self.svmtrain_file_name, self.gnuplot_full_path, self.scaled_file)
        logging.info('Cross validation...')
        f = subprocess.Popen(self.cmd, shell = True, stdout = subprocess.PIPE).stdout

        line = ''
        while True:
            last_line = line
            line = f.readline()
            if not line: break

        c,g,rate = map(float,last_line.split())

        logging.info('Best c={0}, g={1} CV rate={2}'.format(c,g,rate))
        self.perform_svmtrain_training(c,g)

# # cmd = '{0} -svmtrain "{1}" -gnuplot "{2}" "{3}"'.format(grid_py, svmtrain_exe, gnuplot_exe, scaled_file)
# # logging.info('Cross validation...')
# # f = Popen(cmd, shell = True, stdout = PIPE).stdout

# # line = ''
# # while True:
# #     last_line = line
# #     line = f.readline()
# #     if not line: break

# # c,g,rate = map(float,last_line.split())

# # logging.info('Best c={0}, g={1} CV rate={2}'.format(c,g,rate))

    def perform_svmtrain_training(self,c,g):
        self.cmd = '{0} -c {1} -g {2} "{3}" "{4}"'.format(self.svmtrain_file_name,c,g,self.scaled_file,self.model_file)
        logging.info('Training...')
        subprocess.Popen(self.cmd, shell = True, stdout = subprocess.PIPE).communicate()

        if not self.test_path == '':
            logging.info('Output model: {0}'.format(self.model_file))
            self.cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(self.svmscale_file_name, self.range_file, self.test_path, self.scaled_test_file)
            logging.info('Scaling testing data...')
            subprocess.Popen(self.cmd, shell = True, stdout = subprocess.PIPE).communicate()

            self.cmd = '{0} "{1}" "{2}" "{3}"'.format(self.svmpredict_file_name, self.scaled_test_file, self.model_file, self.predict_test_file)
            logging.info('Testing...')
            subprocess.Popen(self.cmd, shell = True).communicate()

            logging.info('Output prediction: {0}'.format(self.predict_test_file))

# # cmd = '{0} -c {1} -g {2} "{3}" "{4}"'.format(svmtrain_exe,c,g,scaled_file,model_file)
# # logging.info('Training...')
# # Popen(cmd, shell = True, stdout = PIPE).communicate()

# # logging.info('Output model: {0}'.format(model_file))
# # if len(sys.argv) > 2:
# #     cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
# #     logging.info('Scaling testing data...')
# #     Popen(cmd, shell = True, stdout = PIPE).communicate()

# #     cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
# #     logging.info('Testing...')
# #     Popen(cmd, shell = True).communicate()

# #     logging.info('Output prediction: {0}'.format(predict_test_file))

    # write a function to read dataframe from file
    # def read_data(self,filename):
    #     data = pd.read_csv(filename, header=None)
    #     data.columns = ['label', 'x1', 'x2']
    #     return data 
def main():
    easy = Easy('','', sys.argv[1], sys.argv[2])
    easy.validate_paths()
    easy.create_train_files_path()
    easy.create_test_files_path()
    easy.perform_svmscale_scaling_training_data()
    easy.perform_grid_py_cross_validation()
    
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        #  python utils\easyClass.py -a 1 -b 2
        logging.info('Usage: {0} training_file [testing_file]'.format(sys.argv[0]))
        raise SystemExit
    main()
