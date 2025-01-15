"""
Filename: classifier.py
Author: Wang Chuyu
Contact: wangchuyu@kongfoo.cn
"""
import pickle
import logging
import warnings
import pandas as pd
import xgboost as xgb
import numpy as np
import xgboost as xgb
import sklearn as sklearn
from sklearn.svm import SVC
from models.model import Model
from models.combiner import Combiner 
from collections import defaultdict
from dataset.probioFeat import KmerSplitter
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_classif, chi2
from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from utils.language_helpers import NgramLM
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Classifier(Model):
    """
    用于分类任务的机器学习模型训练和评估。
    """
    def __init__(self, config):
        self.config = defaultdict(lambda: None, config)
        self.pipeline = None
        self.selectbest = None
        self.scoring = None
        # denoting the scorer that would be used to find the best parameters for refitting the estimator at the end.
        self.refit = "accuracy"
        self.train_with_prob = True

    def read_data(self):
        """
        读取数据集并返回DataFrame格式的特征和标签。
        """
        X = pd.read_csv(self.config['fine_feats'])
        X_train = pd.read_csv(self.config['fine_feat_train'])
        X_test = pd.read_csv(self.config['fine_feat_test'])
        y = pd.read_csv(self.config['labels_train'])
        y_test = pd.read_csv(self.config['labels_test'])
        y_all = pd.read_csv(self.config['labels'])
        self.fine_kmers = X.columns.values
        return X, X_train, X_test, y.values.ravel(), y_test.values.ravel(), y_all.values.ravel()
    
    def create_model(self):
        """
        创建模型对象。
        """
        self.num_folds = self.config['num_folds'] 
        self.num_feats = self.config['num_feats']
        self.svm__C = self.config['svm__C']
        self.svm__gamma = self.config['svm__gamma']
        self.xgb__max_depth = self.config['xgb__max_depth']
        self.xgb__n_estimators = self.config['xgb__n_estimators']
        self.svm__kernel = self.config['svm__kernel']
        self.model_name = self.config['model_name']
        # define evaluation metrics
        def getR2(clf,x,y_real):
            # y_pred = clf.predict(x)
            # r2 = r2_score(y_real, y_pred)
            return "just test"

        self.scoring = {
                "accuracy": 'accuracy',
                #"my_R2":getR2,
                # 'f1': 'f1',
                # 'roc_auc': 'roc_auc'
            }
        if self.model_name == "svm":
            parameters = {
                'kernel': self.svm__kernel,
                'C': self.svm__C,
                'gamma': self.svm__gamma,
                'random_state': [1],
            }
            model = GridSearchCV(SVC(probability=self.train_with_prob), parameters, cv=self.num_folds, n_jobs=self.config['njobs'], 
                                refit=self.refit, scoring=self.scoring)
        else:
            parameters = {
                'max_depth': self.xgb__max_depth,
                'n_estimators': self.xgb__n_estimators,
                'random_state': [1],
            }
            model = GridSearchCV(xgb.XGBClassifier(), parameters, cv=self.num_folds, n_jobs=self.config['njobs'], 
                                refit=self.refit, scoring=self.scoring)

        scaler = None # 标准化m
        if self.config['scaler'] == 'std': 
            scaler = StandardScaler() 
        elif self.config['scaler'] == 'mm':
            scaler = MinMaxScaler()  
        self.selectbest = SelectKBest(k=self.num_feats)  # 特征选择
        self.pipeline = Pipeline([('scaler', scaler), ('model', model)], verbose=self.config['verbose'])

    def get_best_parameter_metric(self, cv_results, metric, best_index):
        """
        extract the evaluation metric of the tuned/best model on each vaildation set
        """
        selected_keys = list(filter(lambda key:'split' in key and metric in key, cv_results))
        # construct matrix, matrix[i][j] represent the jth set parameter score in the ith vaildation set
        matrix = np.concatenate([np.expand_dims(cv_results[key], axis = 0) for key in selected_keys], axis=0)
        return matrix[:, best_index].T.flatten()

    def train_model(self, X, y):
        """
        训练模型并返回交叉验证评估指标结果。

        参数：
            X: 特征数据。
            y: 标签数据。

        返回：
            交叉验证评估指标结果。
        """
        X_sel = self.selectbest.fit(self.X_train, self.y_train).transform(X)
        logging.info('SelectK Best Success...')
        self.pipeline.fit(X_sel, y)
        # scores = cross_validate(self.pipeline, X_sel, y, cv=self.num_folds,
        #                         scoring=('f1', 'accuracy', 'roc_auc'), return_train_score=False)
        gsearch_result = self.pipeline.named_steps['model'].cv_results_
        best_index = self.pipeline.named_steps['model'].best_index_
        # retrieve the metrics scores of the best hyper-parameter
        logging.debug(pd.DataFrame(gsearch_result).style)
        scores = {"test_%s"%metric : self.get_best_parameter_metric(gsearch_result, metric, best_index) for metric in self.scoring.keys()}
        #logging.info(f"Cross validation scores for each split: {scores}")
        grid_file_path = self.config["grid_record"]
        with open(grid_file_path, 'wb') as f:  
            pickle.dump({"search":gsearch_result, "index":best_index}, f)  
        return scores

    def save_model(self):
        """
        保存训练好的模型和特征选择结果。
        """
        relative_path = self.config['model_dir']
        selectid = self.selectbest.get_support()
        with open(relative_path, 'wb') as f:
            pickle.dump(self.pipeline, f)

        core_select = pd.DataFrame([self.fine_kmers[idx] for idx, b in enumerate(selectid) if b ])
        core_select.to_csv(self.config['save_core_select'], index=False)

    def train(self):
        """
        主函数，用于执行整个模型训练过程。
        """
        X, self.X_train, self.X_test, self.y_train, self.y_test, y_all = self.read_data()
        logging.info('Reading Data Success...')
        self.create_model()
        logging.info('Creating Model Success...')
        scores = self.train_model(self.X_train, self.y_train)
        logging.info('Training Model Success...')
        self.save_model()
        # logging.info(f"Avg val acc: {scores['test_accuracy'].mean()}")
        # logging.info(f"Avg val auc: {scores['test_roc_auc'].mean()}")
        print(f"{self.config['num_folds']}-fold validation accuracy: {scores['test_accuracy'].mean()}")
        X_sel = self.selectbest.transform(self.X_test)
        y_hat = self.pipeline.predict(X_sel)
        pred_acc = sklearn.metrics.accuracy_score(self.y_test, y_hat)
        print("y_test",self.y_test)
        print("y_pred",y_hat)
        # my report 
        report = classification_report(self.y_test, y_hat)
        print(report)
        print("-"*40)
        X_train_my = self.selectbest.transform(self.X_train)
        y_hat_train_my = self.pipeline.predict(X_train_my)
        report = classification_report(self.y_train, y_hat_train_my)
        print(report)

        logging.info(f"Testing accuracy: {pred_acc}.")
        scores['pred_acc'] = pred_acc
        try:
            pred_roc = sklearn.metrics.roc_auc_score(self.y_test, y_hat)
            logging.info(f"Testing AUC: {pred_roc}.")
            scores['pred_roc'] = pred_roc
        except ValueError:
            pass
        print(f"Testing accuracy: {pred_acc}.")
        return scores

        # metrics = dataiku.Dataset("metrics")
        # metrics.write_with_schema(pd.DataFrame({
        #     "max_acc": [scores['test_accuracy'].max()],
        #     "avg_acc": [scores['test_accuracy'].mean()],
        #     "auc": [scores['test_roc_auc'].mean()]
        # }))
    
    def _process_gene(self, seq):
        """
        计算每个子序列在基因中出现的频率。
        """
        mers2num = {}
        for it_index in self.mers:
            mers2num[it_index] = 0
        for i in range(len(seq)):
            for j in self.k_mer_set:
                if i + j <= len(seq) and seq[i:i+j] in mers2num:
                    mers2num[seq[i:i+j]] = mers2num[seq[i:i+j]] + 1
        feats = []
        for i, mer in enumerate(self.mers):
            num = mers2num[mer]
            feats.append(num / (len(seq) + 1 - len(mer)))
        return feats

    def read_selected_features(self, k):
        """
        读取选择的特征ID。

        参数：
        - k：子序列长度。

        返回：
        计算结果。

        示例：
        >>> read_selected_features(k)
        """
        # v2_mer_train_select = dataiku.Dataset(f'{k}_mer_train_select_')  # change
        # lines = v2_mer_train_select.get_dataframe()
        v2_mer_train_select = pd.read_csv(f"{self.config['save_path']}{k}{self.config['output_dir_fselect']}")
        lines = v2_mer_train_select.values.tolist()
        feature_id = [int(line[0]) for line in lines]
        return feature_id
    
    def predict(self, genes = None, feats = None, ground_truth = False, preprocesser = None, output_proba = False):
        """
        用于执行根据模型所筛选得到特征进行模型推理的过程。
        参数:
            genes: 待预测的基因序列。 
            
        返回值:
            y_hat: 预测结果。
            pred_acc: 预测准确率。 
        """
        ground_truth = False
        if type(genes) == list and type(ground_truth) == list: 
            assert len(genes) == len(ground_truth)
        self.mers = []
        self.k_mer_set = self.config['k_mer_set']
        for i in self.k_mer_set:
            buf = NgramLM(i,[]).kmers() #self.dfs('', i)
            self.mers.extend(buf)
        
        core_select = pd.read_csv(self.config['save_core_select'])
        if genes and type(genes) == list:
            feats = pd.DataFrame({gene.split(',')[0]:self._process_gene(gene.split(',')[-1]) for gene in genes})
            feats = feats.T # 
            feats.columns = self.mers
        
        core_select = core_select.iloc[:, 0]
        core_select = core_select.tolist()
        # print(set(core_select))
        # print("*"*40)
        # print(set(list(feats.columns)))
        assert(set(core_select).issubset(set(list(feats.columns))))
        X_sel = feats.loc[:,core_select]
        X_sel.to_csv(self.config['core_feat'], index=True)
        relative_path = self.config['model_dir']
        
        with open(relative_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        params = self.pipeline.get_params()

        try:
            assert X_sel.shape[1] == params['scaler'].n_features_in_
            y_hat = self.pipeline.predict(X_sel)
            proba_list = []
            if output_proba:
                y_prob = self.pipeline.predict_proba(X_sel)
                for i,j in zip(y_hat,y_prob):
                    proba_list.append(j[int(i)])
            print(proba_list)
            logging.info(f"Predicted labels: {' '.join(y_hat.astype(str))}")
            if ground_truth: 
                self.y_test = ground_truth if type(ground_truth) == list \
                    else [i[-1] for i in pd.read_csv(ground_truth,index_col=0) \
                        .loc[list(X_sel.index.values)].values]  
                pred_acc = sklearn.metrics.accuracy_score(self.y_test, y_hat) 
                logging.info(f"Prediction accuracy: {pred_acc}.")
                if output_proba != True:
                    pred_roc = sklearn.metrics.roc_auc_score(self.y_test, y_hat)
                    logging.info(f"Prediction AUC: {pred_roc}.")
            else:
                pred_acc = None
            predict_labels = pd.DataFrame(y_hat)
            predict_labels['gene_id'] = feats.index
            predict_labels.set_index('gene_id', inplace=True)
            if len(proba_list) == 0: 
                predict_labels.columns = ['label']
            else:
                predict_labels['proba'] = proba_list
                predict_labels.columns = ['label','proba']
            predict_labels.to_csv(self.config['predict_labels'], index=True)
        except UnboundLocalError as e:
            logging.error(f'UnboundLocalError: {e.message}.')
        except AttributeError as e:
            logging.error(f'AttributeError: {e.message}.')
        if output_proba:
            return y_prob, pred_acc
        return y_hat, pred_acc
    
            
# 调用方法
# if __name__ == "__main__":
#     # num_feats = int(dataiku.get_custom_variables()["num_feats"])
#     # model_name = dataiku.get_custom_variables()["model"]
#     num_feats = 64
#     model_name = 'svm'

#     classifier = Classifier(num_feats, model_name)
#     scores = classifier.main()

#     # metrics = dataiku.Dataset("metrics")
#     # metrics.write_with_schema(pd.DataFrame({
#     #     "max_acc": [scores['test_accuracy'].max()],
#     #     "avg_acc": [scores['test_accuracy'].mean()],
#     #     "auc": [scores['test_roc_auc'].mean()]
#     # }))
#     metrics = pd.DataFrame({
#         "max_acc": [scores['test_accuracy'].max()],
#         "avg_acc": [scores['test_accuracy'].mean()],
#         "auc": [scores['test_roc_auc'].mean()]
#     })
#     metrics.to_csv('test/metrics.csv', index=False)

#     # scores_dd = dataiku.Dataset("scores")
#     # scores_dd.write_with_schema(pd.DataFrame(scores))
#     scores = pd.DataFrame(scores)
