import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report,roc_auc_score,multilabel_confusion_matrix,recall_score,precision_score,f1_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')
import joblib
from Code.options import parse_option


def save_test_metrics_to_csv(metrics_dict, model_name, output_file):
    try:
        # 确保输入参数的类型正确
        if not isinstance(metrics_dict, dict):
            raise TypeError("metrics_dict 必须是字典类型")
        if not isinstance(model_name, str):
            raise TypeError("model_name 必须是字符串类型")
        if not isinstance(output_file, str):
            raise TypeError("output_file 必须是字符串类型")

        # 获取所有以'test_'开头的键
        test_keys = [key for key in metrics_dict.keys() if key.startswith('test_')]

        # 检查是否找到了测试指标
        if not test_keys:
            raise ValueError("未找到以'test_'开头的指标")

        # 检查所有测试指标的数组长度是否一致
        array_lengths = [len(metrics_dict[key]) for key in test_keys]
        if len(set(array_lengths)) != 1:
            raise ValueError("所有测试指标的数组长度必须相同")

        n_folds = len(metrics_dict[test_keys[0]])

        # 创建数据列表
        data = []
        for fold_idx in range(n_folds):
            row = {
                'model_name': model_name,
                'fold': f'fold_{fold_idx + 1}'
            }
            # 添加每个测试指标的值
            for key in test_keys:
                metric_name = key.replace('test_', '')
                try:
                    row[metric_name] = float(metrics_dict[key][fold_idx])
                except (TypeError, ValueError) as e:
                    raise ValueError(f"无法将指标 {key} 的值转换为浮点数: {str(e)}")
            data.append(row)

        # 创建DataFrame并保存到CSV
        df = pd.DataFrame(data)
        try:
            df.to_csv(output_file, index=False)
            print(f"结果已成功保存到 {output_file}")
        except Exception as e:
            raise IOError(f"保存CSV文件时出错: {str(e)}")

    except Exception as e:
        print(f"错误: {str(e)}")
        raise  # 重新抛出异常，便于调试

def smote_sampler(X_train, y_train, strategy='smote'):
    if(strategy == 'smote'):
        ## smote 2
        oversample = SMOTE(k_neighbors=3)
        X_train2 ,y_train2 = oversample.fit_resample(X_train ,y_train)
        counter = Counter(y_train2)
    elif(strategy == 'mix'):
        ## smote under Mix
        smote_strategy={'C':20, 'S':20, 'F':22}
        over = SMOTE(sampling_strategy=smote_strategy,k_neighbors=3)
        under_strategy={'C':20, 'S':20, 'F':11}
        under = RandomUnderSampler(sampling_strategy=under_strategy)
        steps = [('o', over), ('u', under)]
        pipeline_o_v = imblearn.pipeline.Pipeline(steps=steps)
        # transform the dataset
        X_train2 ,y_train2 = pipeline_o_v.fit_resample(X_train, y_train)
        # summarize the new class distribution
        counter = Counter(y_train2)
    elif(strategy == 'none'):
        counter = Counter(y_train)
    #return counter
    return X_train2 ,y_train2

def mean_std(scores):
    m = scores.mean()
    s = scores.std()
    res = m+s
    return res

def metric_model(y_test,y_pred,average = 'macro'):
    accuracy = accuracy_score(y_test, y_pred, average='macro')
    precision= precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    AUC = roc_auc_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1, AUC

def scores_extract_metrics(scores=None,model_name=None,modality_name=None,reduction_method = None):
    if not scores or not model_name:
        print("Not all required arguments provided, function not executed.")
        return None
    m = pd.DataFrame(scores).iloc[:,2:].mean()
    s = pd.DataFrame(scores).iloc[:,2:].std()
    res = pd.concat([m,s],axis=1, join='inner',keys=['mean','std'])
    res = res.sort_index(axis=1,ascending=False)
    combined = pd.DataFrame(res['mean'].round(4).astype(str) + "±" + res['std'].round(4).astype(str))
    combined.columns = [model_name]
    #combined.sort_index(axis=0, ascending=False)
    return combined.T

def scores_extract_metrics_with_estimator(scores=None,model_name=None,modality_name=None,reduction_method = None):
    if modality_name==None:
        print("modality_name is None,please check the modality_name argument,eg, 'Dose'.")
        return None
    if not scores or not model_name:
        print("Not all required arguments provided, function not executed.")
        return None
    opt = parse_option(print_option=False)
    estimators = scores['estimator']
    if opt.need_clinical_info:
        modality = modality_name + '+clinical'
    else:
        modality = modality_name
    if opt.need_classifier_optimization:
        model_dir = os.path.join('./model_save',modality,'function' + str(opt.function_threshold), 'with_ml_classification',reduction_method)
        if not os.path.exists(model_dir ):
            os.makedirs(model_dir)
    else:
        model_dir  = os.path.join('./model_save',modality,'function' + str(opt.function_threshold), 'without_ml_classification',reduction_method)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    for idx, estimator in enumerate(estimators, 1):
        # 保存模型
        model_path = os.path.join(model_dir,f'{model_name}_model_cv_{idx}.joblib')
        joblib.dump(estimator,model_path)
    different_test_fold_path = os.path.join(model_dir,f'{model_name}_cross_valid_result.csv')
    # 生成不同fold的test结果
    try:
        save_test_metrics_to_csv(scores,model_name,different_test_fold_path)
    except Exception as e:
        print(f"程序执行失败: {str(e)}")

    m = pd.DataFrame(scores).iloc[:,3:].mean()
    s = pd.DataFrame(scores).iloc[:,3:].std()
    res = pd.concat([m,s],axis=1, join='inner',keys=['mean','std'])
    res = res.sort_index(axis=1,ascending=False)
    combined = pd.DataFrame(res['mean'].round(4).astype(str) + "±" + res['std'].round(4).astype(str))
    combined.columns = [model_name]
    #combined.sort_index(axis=0, ascending=False)
    return combined.T

def model_selection(X_train, y_train, scoring):
    names = [
        "Nearest Neighbors",
        #"Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        "Logistic Regression",
        "GBM",
        "XGBoost",
        "lightGBM",
        "CatBoost",
    ]
    classifiers = [
        KNeighborsClassifier(3,n_jobs=12),
        #SVC(kernel="linear", C=0.025, probability=True),
        SVC(gamma=2, C=1,probability=True),
        GaussianProcessClassifier(1.0 * RBF(1.0),multi_class="one_vs_rest",n_jobs=12),
        DecisionTreeClassifier(max_depth=5,criterion="gini"),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1,n_jobs=12),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression(multi_class="multinomial",n_jobs=12),
        GradientBoostingClassifier(),
        xgb.XGBClassifier(tree_method="hist", enable_categorical=True),
        lgb.LGBMClassifier(),
        cb.CatBoostClassifier(),
    ]
    #scoring = {'AUC': 'roc_auc_ovr', 'Accuracy': make_scorer(accuracy_score),'precision':make_scorer(precision_score(average='macro')),'recall':'recall_macro','f1':make_scorer(f1_score(average='macro'))}
    performance = pd.DataFrame()
    for name, model in zip(names, classifiers):
        print(name)
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=1)
        if(name == 'XGBoost'):
            y2 = LabelEncoder().fit_transform(y_train).astype('int64')
            x2 = X_train.astype('category')
            scores = cross_validate(model, x2, y2, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise', return_train_score=True)
            res = scores_extract_metrics(scores, name)
            performance = pd.concat([performance, res], axis=0)
        elif(name == 'lightGBM'):
            x2 = X_train.astype('category')
            scores = cross_validate(model, x2, y_train, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise', return_train_score=True)
            res = scores_extract_metrics(scores, name)
            performance = pd.concat([performance, res], axis=0)
        else:
            scores = cross_validate(model, X_train, y_train, scoring=scoring , cv=cv, n_jobs=-1, return_estimator=True, return_train_score=True)
            res = scores_extract_metrics(scores, name)
            performance = pd.concat([performance, res], axis=0)
    performance = performance.sort_index(axis=1,ascending=False)
    return performance


def model_cv(X_train, y_train, scoring, name = "RBF SVM"):
    names = [
        "Nearest Neighbors",
        #"Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        "Logistic Regression",
        "GBM",
        "XGBoost",
        "lightGBM",
        "CatBoost",
    ]
    classifiers = [
        KNeighborsClassifier(3,n_jobs=12),
        #SVC(kernel="linear", C=0.025, probability=True),
        SVC(gamma=2, C=1,probability=True),
        GaussianProcessClassifier(1.0 * RBF(1.0),multi_class="one_vs_rest",n_jobs=12),
        DecisionTreeClassifier(max_depth=5,criterion="gini"),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1,n_jobs=12),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression(multi_class="multinomial",n_jobs=12),
        GradientBoostingClassifier(),
        xgb.XGBClassifier(tree_method="hist", enable_categorical=True),
        lgb.LGBMClassifier(),
        cb.CatBoostClassifier(),
    ]
    #scoring = {'AUC': 'roc_auc_ovr', 'Accuracy': make_scorer(accuracy_score),'precision':make_scorer(precision_score(average='macro')),'recall':'recall_macro','f1':make_scorer(f1_score(average='macro'))}
    index = names.index(name)
    model= classifiers[index]
    performance = pd.DataFrame()
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    if(name == 'XGBoost'):
        y2 = LabelEncoder().fit_transform(y_train).astype('int64')
        x2 = X_train.astype('category')
        scores = cross_validate(model, x2, y2, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise', return_train_score=True)
        res = scores_extract_metrics(scores, name)
        performance = pd.concat([performance, res], axis=0)
    elif(name == 'lightGBM'):
        x2 = X_train.astype('category')
        scores = cross_validate(model, x2, y_train, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise', return_train_score=True)
        res = scores_extract_metrics(scores, name)
        performance = pd.concat([performance, res], axis=0)
    else:
        scores = cross_validate(model, X_train, y_train, scoring=scoring , cv=cv, n_jobs=-1, return_estimator=True, return_train_score=True)
        res = scores_extract_metrics(scores, name)
        performance = pd.concat([performance, res], axis=0)
    performance = performance.sort_index(axis=1,ascending=False)
    return performance

def merge_performance_results(folder_path ):
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('Performance_') and f.endswith('.csv')]
    merged_data = pd.DataFrame()
    for file_name in csv_files:
        # 读取CSV文件
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path,sep='\t')
        # 添加文件名列
        df['fs_type'] = file_name
        # 合并数据到merged_data中
        merged_data = pd.concat([merged_data, df], ignore_index=True)
    return merged_data





#model = xgb.XGBClassifier(tree_method="hist",enable_categorical=True)
# model= cb.CatBoostClassifier(verbose=0, n_estimators=100)
# cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
# y2=LabelEncoder().fit_transform(y_train).astype('int64')
# x2= X_train.astype('category')
# n_scores = cross_val_score(model, x2, y2, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# scores = cross_validate(model, x2, y2, scoring=scoring , cv=cv, n_jobs=-1,  error_score='raise', return_train_score=True)
