import pandas as pd
import numpy as np
import os
# from sklearn.feature_selection import RFECV
# from sklearn.svm import SVC
# #import my_util.Features_selection as Fs
# from utils.Classifier_construction_sklearn_parallel import scores_extract_metrics,scores_extract_metrics_with_estimator
from sklearn.metrics import make_scorer
# from sklearn.metrics import precision_score,f1_score,accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import RidgeClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from options import parse_option
from utils import load_data
# import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# noinspection PyShadowingNames
def model_selection(X_train, y_train, X_test, y_test,test_data, modality_name,
                   target_name=None, reduction_method=None, need_predict_score=False):
    # 函数参数创建新的局部作用域，不会隐藏外部变量
    # pylint: disable=redefined-outer-name
    
    def check_overfitting(train_accuracy, val_accuracy, model_name):
        """检查并警告可能的过拟合：train accuracy高但validation accuracy低"""
        gap = train_accuracy - val_accuracy
        if train_accuracy >= 0.95 and gap > 0.15:
            print(f"  ⚠️  Warning: Possible overfitting! Train: {train_accuracy:.4f}, CV: {val_accuracy:.4f}, Gap: {gap:.4f}")
        elif train_accuracy >= 0.90 and gap > 0.10:
            print(f"  ⚠️  Caution: Potential overfitting! Train: {train_accuracy:.4f}, CV: {val_accuracy:.4f}, Gap: {gap:.4f}")
    names = [
        "Nearest Neighbors",
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
        KNeighborsClassifier(3),
        GaussianProcessClassifier(1.0 * RBF(1.0), multi_class="one_vs_rest", n_jobs=4),
        DecisionTreeClassifier(max_depth=5, criterion="gini"),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=4),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(reg_param=0.1),  # 添加正则化处理共线性
        LogisticRegression(multi_class="multinomial", n_jobs=4, max_iter=10000),
        GradientBoostingClassifier(),
        xgb.XGBClassifier(tree_method="hist", enable_categorical=True, eval_metric='mlogloss'),
        lgb.LGBMClassifier(verbosity=-1),
        cb.CatBoostClassifier(silent=True),
    ]

    if target_name is not None and target_name in names:
        idx = names.index(target_name)
        names = [names[idx]]
        classifiers = [classifiers[idx]]

    performance = []
    predictions_all = []

    for name, model in zip(names, classifiers):
        print(f"Running: {name}")
        # 针对KNN、DT、RF，做参数调优
        if name == "Nearest Neighbors":
            # 对于小数据集，k值范围适当缩小
            k_range = range(3, 12)  # k: 3-11
            best_score = -1
            best_k = 3
            # 使用交叉验证进行参数调优
            n_splits = 10
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                # 使用交叉验证评估
                scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                score = scores.mean()
                if score > best_score:
                    best_score = score
                    best_k = k
            best_model = KNeighborsClassifier(n_neighbors=best_k)
            best_model.fit(X_train, y_train)
            # 使用交叉验证评估最终模型
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            val_accuracy = cv_scores.mean()
            val_std = cv_scores.std()
            # 计算train accuracy
            y_train_pred = best_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
            print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
            check_overfitting(train_accuracy, val_accuracy, name)

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy ({name}): {test_accuracy:.4f}")

            auc_val = roc_auc_score(y_test, y_proba[:, 1], multi_class="ovr") if len(np.unique(y_test)) > 1 else np.nan
            res = {
                "Classifier": "Nearest Neighbors",
                "Best_k": best_k,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "AUC": auc_val
            }
            performance.append(res)
            if need_predict_score:
                prob_1 = y_proba[:, 1] if y_proba is not None and y_proba.shape[1] > 1 else y_proba[:, 0]
                pred_df = pd.DataFrame({
                    'ID': test_data['ID'].values,
                    'predicted_label': y_pred,
                    'prob_1': prob_1,
                    'true_label': y_test.values if hasattr(y_test, "values") else y_test,
                    'Classifier': name
                })
                predictions_all.append(pred_df)
            continue

        if name == "Decision Tree":
            # 对于小数据集，限制树深度避免过拟合
            k_range = range(3, 9)  # max_depth: 3-8
            best_score = -1
            best_k = 3
            # 使用交叉验证进行参数调优
            n_splits = 10
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for k in k_range:
                # 添加防止过拟合的参数：min_samples_split, min_samples_leaf
                dt = DecisionTreeClassifier(max_depth=k, min_samples_split=5, min_samples_leaf=2, random_state=42)
                # 使用交叉验证评估
                scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                score = scores.mean()
                if score > best_score:
                    best_score = score
                    best_k = k
            best_model = DecisionTreeClassifier(max_depth=best_k, min_samples_split=5, min_samples_leaf=2, random_state=42)
            best_model.fit(X_train, y_train)
            # 使用交叉验证评估最终模型
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            val_accuracy = cv_scores.mean()
            val_std = cv_scores.std()
            # 计算train accuracy
            y_train_pred = best_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
            print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
            check_overfitting(train_accuracy, val_accuracy, name)

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy ({name}): {test_accuracy:.4f}")
            auc_val = roc_auc_score(y_test, y_proba[:, 1], multi_class="ovr") if len(np.unique(y_test)) > 1 else np.nan
            res = {
                "Classifier": "Decision Tree",
                "Best_depth": best_k,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "AUC": auc_val
            }
            performance.append(res)
            if need_predict_score:
                prob_1 = y_proba[:, 1] if y_proba is not None and y_proba.shape[1] > 1 else y_proba[:, 0]
                pred_df = pd.DataFrame({
                    'ID': test_data['ID'].values,
                    'predicted_label': y_pred,
                    'prob_1': prob_1,
                    'true_label': y_test.values if hasattr(y_test, "values") else y_test,
                    'Classifier': name
                })
                predictions_all.append(pred_df)
            continue

        if name == "Random Forest":
            # 对于小数据集，使用较少的树数量
            n_range = [10, 20, 30, 50]
            best_score = -1
            best_n = 10
            # 使用交叉验证进行参数调优
            n_splits = 10
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for n in n_range:
                # 添加防止过拟合的参数：限制深度、min_samples_split、max_features
                rf = RandomForestClassifier(n_estimators=n, max_depth=5, min_samples_split=5, 
                                          min_samples_leaf=2, max_features='sqrt', n_jobs=4, random_state=42)
                # 使用交叉验证评估
                scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                score = scores.mean()
                if score > best_score:
                    best_score = score
                    best_n = n
            best_model = RandomForestClassifier(n_estimators=best_n, max_depth=5, min_samples_split=5,
                                               min_samples_leaf=2, max_features='sqrt', n_jobs=4, random_state=42)
            best_model.fit(X_train, y_train)
            # 使用交叉验证评估最终模型
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            val_accuracy = cv_scores.mean()
            val_std = cv_scores.std()
            # 计算train accuracy
            y_train_pred = best_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
            print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
            check_overfitting(train_accuracy, val_accuracy, name)
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy ({name}): {test_accuracy:.4f}")
            auc_val = roc_auc_score(y_test, y_proba[:, 1], multi_class="ovr") if len(np.unique(y_test)) > 1 else np.nan
            res = {
                "Classifier": "Random Forest",
                "Best_n_estimators": best_n,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "AUC": auc_val
            }
            performance.append(res)
            if need_predict_score:
                prob_1 = y_proba[:, 1] if y_proba is not None and y_proba.shape[1] > 1 else y_proba[:, 0]
                pred_df = pd.DataFrame({
                    'ID': test_data['ID'].values,
                    'predicted_label': y_pred,
                    'prob_1': prob_1,
                    'true_label': y_test.values if hasattr(y_test, "values") else y_test,
                    'Classifier': name
                })
                predictions_all.append(pred_df)
            continue

        # 其它模型，使用验证集进行参数调优
        try:
            # 需要调参的模型列表
            models_need_tuning = ["Logistic Regression", "XGBoost", "lightGBM", "CatBoost", "GBM", "AdaBoost", "Neural Net"]
            
            # 使用交叉验证进行参数调优和评估
            n_splits = 10
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # 根据模型类型进行参数调优
            if name == "Logistic Regression":
                # Logistic Regression 调参：C值（使用更强的正则化防止过拟合）
                C_range = [0.001, 0.01, 0.1, 1, 10]  # 移除100，使用更强的正则化
                best_score = -1
                best_C = 1
                for C in C_range:
                    lr = LogisticRegression(C=C, multi_class="multinomial", n_jobs=4, max_iter=10000, 
                                          penalty='l2', random_state=42)
                    # 使用交叉验证评估
                    scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                    score = scores.mean()
                    if score > best_score:
                        best_score = score
                        best_C = C
                best_model = LogisticRegression(C=best_C, multi_class="multinomial", n_jobs=4, max_iter=10000,
                                               penalty='l2', random_state=42)
                best_model.fit(X_train, y_train)
                # 使用交叉验证评估最终模型
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                val_accuracy = cv_scores.mean()
                val_std = cv_scores.std()
                # 计算train accuracy
                y_train_pred = best_model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                print(f"Best C for {name}: {best_C}")
                print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
                print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
                check_overfitting(train_accuracy, val_accuracy, name)
            
            elif name == "XGBoost":
                # XGBoost 调参：n_estimators (小数据集使用较小的值)
                n_range = [10, 20, 30, 50]
                best_score = -1
                best_n = 20
                for n in n_range:
                    # 添加防止过拟合的参数：限制深度、正则化
                    xgb_model = xgb.XGBClassifier(n_estimators=n, max_depth=3, learning_rate=0.1,
                                                 reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
                                                 tree_method="hist", enable_categorical=True,
                                                 eval_metric='mlogloss', random_state=42)
                    # 使用交叉验证评估
                    scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                    score = scores.mean()
                    if score > best_score:
                        best_score = score
                        best_n = n
                best_model = xgb.XGBClassifier(n_estimators=best_n, max_depth=3, learning_rate=0.1,
                                               reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
                                               tree_method="hist", enable_categorical=True,
                                               eval_metric='mlogloss', random_state=42)
                best_model.fit(X_train, y_train)
                # 使用交叉验证评估最终模型
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                val_accuracy = cv_scores.mean()
                val_std = cv_scores.std()
                # 计算train accuracy
                y_train_pred = best_model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                print(f"Best n_estimators for {name}: {best_n}")
                print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
                print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
                check_overfitting(train_accuracy, val_accuracy, name)
            
            elif name == "lightGBM":
                # LightGBM 调参：n_estimators (小数据集使用较小的值)
                n_range = [10, 20, 30, 50]
                best_score = -1
                best_n = 20
                for n in n_range:
                    # 添加防止过拟合的参数：限制深度、正则化
                    lgb_model = lgb.LGBMClassifier(n_estimators=n, max_depth=3, learning_rate=0.1,
                                                  reg_alpha=0.1, reg_lambda=1.0, min_child_samples=5,
                                                  verbosity=-1, random_state=42)
                    # 使用交叉验证评估
                    scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                    score = scores.mean()
                    if score > best_score:
                        best_score = score
                        best_n = n
                best_model = lgb.LGBMClassifier(n_estimators=best_n, max_depth=3, learning_rate=0.1,
                                               reg_alpha=0.1, reg_lambda=1.0, min_child_samples=5,
                                               verbosity=-1, random_state=42)
                best_model.fit(X_train, y_train)
                # 使用交叉验证评估最终模型
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                val_accuracy = cv_scores.mean()
                val_std = cv_scores.std()
                # 计算train accuracy
                y_train_pred = best_model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                print(f"Best n_estimators for {name}: {best_n}")
                print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
                print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
                check_overfitting(train_accuracy, val_accuracy, name)
            
            elif name == "CatBoost":
                # CatBoost 调参：iterations (小数据集使用较小的值)
                n_range = [10, 20, 30, 50]
                best_score = -1
                best_n = 20
                for n in n_range:
                    # 添加防止过拟合的参数：限制深度、正则化、学习率
                    cb_model = cb.CatBoostClassifier(iterations=n, max_depth=3, learning_rate=0.1,
                                                    l2_leaf_reg=3, min_data_in_leaf=5, silent=True, random_state=42)
                    # 使用交叉验证评估
                    scores = cross_val_score(cb_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                    score = scores.mean()
                    if score > best_score:
                        best_score = score
                        best_n = n
                best_model = cb.CatBoostClassifier(iterations=best_n, max_depth=3, learning_rate=0.1,
                                                  l2_leaf_reg=3, min_data_in_leaf=5, silent=True, random_state=42)
                best_model.fit(X_train, y_train)
                # 使用交叉验证评估最终模型
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                val_accuracy = cv_scores.mean()
                val_std = cv_scores.std()
                # 计算train accuracy
                y_train_pred = best_model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                print(f"Best iterations for {name}: {best_n}")
                print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
                print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
                check_overfitting(train_accuracy, val_accuracy, name)
            
            elif name == "GBM":
                # Gradient Boosting 调参：n_estimators (小数据集使用较小的值)
                n_range = [10, 20, 30, 50]
                best_score = -1
                best_n = 20
                for n in n_range:
                    # 添加防止过拟合的参数：限制深度、学习率、min_samples_split
                    gbm_model = GradientBoostingClassifier(n_estimators=n, max_depth=3, learning_rate=0.1,
                                                           min_samples_split=5, min_samples_leaf=2, random_state=42)
                    # 使用交叉验证评估
                    scores = cross_val_score(gbm_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                    score = scores.mean()
                    if score > best_score:
                        best_score = score
                        best_n = n
                best_model = GradientBoostingClassifier(n_estimators=best_n, max_depth=3, learning_rate=0.1,
                                                       min_samples_split=5, min_samples_leaf=2, random_state=42)
                best_model.fit(X_train, y_train)
                # 使用交叉验证评估最终模型
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                val_accuracy = cv_scores.mean()
                val_std = cv_scores.std()
                # 计算train accuracy
                y_train_pred = best_model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                print(f"Best n_estimators for {name}: {best_n}")
                print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
                print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
                check_overfitting(train_accuracy, val_accuracy, name)
            
            elif name == "AdaBoost":
                # AdaBoost 调参：n_estimators (小数据集使用较小的值)
                n_range = [10, 20, 30, 50]
                best_score = -1
                best_n = 10
                for n in n_range:
                    # 添加防止过拟合的参数：降低学习率
                    ab_model = AdaBoostClassifier(n_estimators=n, learning_rate=0.5, random_state=42)
                    # 使用交叉验证评估
                    scores = cross_val_score(ab_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                    score = scores.mean()
                    if score > best_score:
                        best_score = score
                        best_n = n
                best_model = AdaBoostClassifier(n_estimators=best_n, learning_rate=0.5, random_state=42)
                best_model.fit(X_train, y_train)
                # 使用交叉验证评估最终模型
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                val_accuracy = cv_scores.mean()
                val_std = cv_scores.std()
                # 计算train accuracy
                y_train_pred = best_model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                print(f"Best n_estimators for {name}: {best_n}")
                print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
                print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
                check_overfitting(train_accuracy, val_accuracy, name)
            
            elif name == "Neural Net":
                # Neural Net 调参：alpha值（L2正则化，防止过拟合）
                alpha_range = [0.001, 0.01, 0.1, 1, 10]  # 增加更强的正则化选项
                best_score = -1
                best_alpha = 1
                for alpha in alpha_range:
                    # 添加防止过拟合的参数：增加hidden_layer_sizes限制，early_stopping
                    nn_model = MLPClassifier(alpha=alpha, hidden_layer_sizes=(50,), max_iter=1000,
                                            early_stopping=True, validation_fraction=0.1, random_state=42)
                    # 使用交叉验证评估
                    scores = cross_val_score(nn_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                    score = scores.mean()
                    if score > best_score:
                        best_score = score
                        best_alpha = alpha
                best_model = MLPClassifier(alpha=best_alpha, hidden_layer_sizes=(50,), max_iter=1000,
                                          early_stopping=True, validation_fraction=0.1, random_state=42)
                best_model.fit(X_train, y_train)
                # 使用交叉验证评估最终模型
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                val_accuracy = cv_scores.mean()
                val_std = cv_scores.std()
                # 计算train accuracy
                y_train_pred = best_model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                print(f"Best alpha for {name}: {best_alpha}")
                print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
                print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
                check_overfitting(train_accuracy, val_accuracy, name)
            
            else:
                # 对于不需要调参的模型（Gaussian Process, Naive Bayes, QDA），使用原始模型
                if name == "QDA":
                    # QDA对共线性敏感，使用正则化参数
                    best_model = QuadraticDiscriminantAnalysis(reg_param=0.1)
                else:
                    best_model = model
                # 使用整个训练集训练模型
                best_model.fit(X_train, y_train)
                # 使用交叉验证评估
                n_splits = 10
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                val_accuracy = cv_scores.mean()
                val_std = cv_scores.std()
                # 计算train accuracy
                y_train_pred = best_model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                print(f"Train Accuracy ({name}): {train_accuracy:.4f}")
                print(f"CV Accuracy ({name}): {val_accuracy:.4f} (+/- {val_std:.4f})")
                check_overfitting(train_accuracy, val_accuracy, name)

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy ({name}): {test_accuracy:.4f}")
            auc_val = roc_auc_score(y_test, y_proba[:, 1], multi_class="ovr") if (
                        y_proba is not None and len(np.unique(y_test)) > 1) else np.nan
            res = {
                "Classifier": name,
                "Accuracy": test_accuracy,
                "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "AUC": auc_val
            }
            performance.append(res)
            if need_predict_score:
                prob_1 = y_proba[:, 1] if y_proba is not None and y_proba.shape[1] > 1 else y_proba[:, 0]
                pred_df = pd.DataFrame({
                    'ID': test_data['ID'].values,
                    'predicted_label': y_pred,
                    'prob_1': prob_1,
                    'true_label': y_test.values if hasattr(y_test, "values") else y_test,
                    'Classifier': name
                })
                predictions_all.append(pred_df)
        except Exception as e:
            print(f"Model {name} failed: {e}")

    performance_df = pd.DataFrame(performance)
    predictions_df = pd.concat(predictions_all, ignore_index=True) if need_predict_score else pd.DataFrame()
    return performance_df, predictions_df

# 从 CSV 文件中读取数据
def read_id_list(txt_path):
    with open(txt_path) as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == '__main__':
    opt = parse_option(print_option=True)
    # modalitylist = ["PTV-HFL-RD"]
    # modalitylist = ['PTV-HV-RD']
    need_train = True
    modalitylist = ['PTV-HFL-RD',"HFL-RD","PTV-RD", "LFL-R",'PTV-HV-RD',"LFL-D","LFL-RD","PTV-HQ-RD","GTV-RD",
               'WL-D','WL-R','WL-RD','HFL-R', "HFL-D"]
    # 'Peri-GTV_Dose_Ventilation','Dose_GTV','Dose_Peri-GTV']
    # modalitylist = ['Peri-HV-RD','Peri-HP-RD']
    # modalitylist = ['ALL']
    for i in modalitylist:
        opt.modality = i
        # modality = 'dose'
        # modality =opt.modality + '+clinical'
        # modality = 'All'
        # modality = 'Peri-GTV+clinical'
        # sampler_choice = ['smote', 'mix', 'none']
        # RFE_choice = ['RFE', 'RFECV']
        # Corr_m = ['pearson', 'spearman']
        # Corr_threshold = [0.6, 0.7, 0.8]
        possible_filenames = [
            f"{opt.modality}_merged_data_part_1_normalized.csv",
            f"{opt.modality}_merged_data_normalized.csv"
        ]
        file_list = []
        for filename in possible_filenames:
            potential_path = os.path.join('./Parallel/PMB/', filename)
            if os.path.exists(potential_path):
                file_list.append(potential_path)
                break

        opt.data_path = file_list
        print('Data path : %s' % opt.data_path)
        data_df = load_data(file_list)

        is_nan = data_df.isna()

        nan_locations = np.where(is_nan)

        for row, col in zip(nan_locations[0], nan_locations[1]):
            print(f"NaN value found at row {row}, column {col}")

        zero_columns = data_df.columns[(data_df == 0).all()].tolist()
        if zero_columns:
            print("Columns with all zeros:", zero_columns)
            sift_input = data_df.drop(columns=zero_columns)

        columns_to_exclude = [
            'Age', 'Dose', 'Gender_1', 'Gender_2',
            'Smoke_1', 'Smoke_2', 'Treatment_0', 'Treatment_1', 'Treatment_2', 'Grade'
        ]

        if opt.need_clinical_info:
            # 只排除'Grade'
            exclude_columns = ['Grade']
            outdir = os.path.join('./Parallel/PMB/multiomics_result', opt.modality)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
        else:
            print('do not need clinical information')
            # 排除这10列
            exclude_columns = columns_to_exclude
            outdir = os.path.join('./Parallel/PMB/multiomics_result', opt.modality)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
        # if opt.need_classifier_optimization:
        #     outdir = os.path.join(outdir, 'with_ml_classification')
        #     if not os.path.exists(outdir):
        #         os.makedirs(outdir)
        # else:
        #     outdir = os.path.join(outdir, 'without_ml_classification')
        #     if not os.path.exists(outdir):
        #         os.makedirs(outdir)

        feature_selected_method = [
            'lasso','fscore','mi'
        ]

        #additional_columns = ['Grade']
        n_folds = 10
        all_performance = []
        all_oof_preds = []
        for fold in range(1, n_folds + 1):
            for method in feature_selected_method:
                selected_train_case_txt_path = os.path.join('./dataset/Dataset_split/PMB',f'Parallel/{n_folds}-fold',
                                                      opt.modality,f'f_ratio_{opt.feature_ratio}_cor_{opt.corr_threshold}',
                                                      f'fold_{fold}_train.txt')
                selected_test_case_txt_path = os.path.join('./dataset/Dataset_split/PMB', f'Parallel/{n_folds}-fold',
                                                            opt.modality,
                                                            f'f_ratio_{opt.feature_ratio}_cor_{opt.corr_threshold}',
                                                            f'fold_{fold}_test.txt')

                train_id_list = read_id_list(selected_train_case_txt_path)
                test_id_list = read_id_list(selected_test_case_txt_path)

                train_data = data_df[data_df['ID'].astype(str).isin(train_id_list)].copy()
                test_data = data_df[data_df['ID'].astype(str).isin(test_id_list)].copy()

                train_label = train_data['Grade']
                test_label = test_data['Grade']

                selected_features_path = os.path.join('./dataset/Dataset_split/PMB',f'Parallel/{n_folds}-fold',
                                                      opt.modality,f'f_ratio_{opt.feature_ratio}_cor_{opt.corr_threshold}',
                                                      f'fold_{fold}_selected_{method}_features_with_info.csv')
                selected_features = pd.read_csv(selected_features_path)
                selected_features = selected_features[selected_features.columns[0]].tolist()

                if len(selected_features) > 30:
                    selected_features = selected_features[:30]

                # 保证train/test特征列一致且不包含被排除列（如ID, Grade等）
                columns_to_remove = [col for col in exclude_columns if col in selected_features]
                use_features = [f for f in selected_features if f not in columns_to_remove]

                X_train = train_data[use_features]
                X_test = test_data[use_features]
                y_train = train_label
                y_test = test_label
                y_test = np.ravel(y_test)
                # X_test = test_data.drop(columns=exclude_columns)
                scoring = {'AUC': 'roc_auc_ovr', 'Accuracy': make_scorer(accuracy_score),
                           'precision': make_scorer(precision_score, average='macro', zero_division=0), 
                           'recall': 'recall_macro',
                           'f1': make_scorer(f1_score, average='macro', zero_division=0)}

                outfile = method + "_performance.csv"
                prob_file = method + "_probability.csv"

                if need_train:
                    performance, predictions_df = model_selection(X_train, y_train, X_test, y_test,test_data,
                                                                  opt.modality,need_predict_score=True)

                    # performance.to_csv(os.path.join(outdir, outfile), encoding='utf-16', sep='\t', index=True,
                    #                    na_rep='NULL')
                    performance['fold'] = fold
                    performance['method'] = method
                    all_performance.append(performance)
                    predictions_df['fold'] = fold
                    predictions_df['method'] = method
                    all_oof_preds.append(predictions_df)
                else:
                    performance = pd.read_csv(os.path.join(outdir, outfile),
                                              encoding='utf-16',
                                              sep='\t',
                                              na_values='NULL',
                                              index_col=0)

        all_perf_df = pd.concat(all_performance, ignore_index=True)
        all_perf_df.to_csv(os.path.join(outdir, "all_folds_performance.csv"), encoding='utf-16', sep='\t', index=False)

        # 计算每个method+Classifier的AUC均值和std
        summary = all_perf_df.groupby(['method', 'Classifier'])['AUC'].agg(['mean', 'std']).reset_index()
        summary.to_csv(os.path.join(outdir, "summary_performance.csv"), encoding='utf-16', sep='\t', index=False)

        # 选出AUC均值最高的method+classifier
        best_row = summary.loc[summary['mean'].idxmax()]
        best_method = best_row['method']
        best_classifier = best_row['Classifier']
        print('AUC均值最高的组合:', best_method, best_classifier)

        # 合并所有fold的预测概率
        all_oof_df = pd.concat(all_oof_preds, ignore_index=True)
        all_oof_df.to_csv(os.path.join(outdir, "all_oof_predictions.csv"), encoding='utf-16', sep='\t', index=False)

        # 只保留最佳method+classifier的oof预测
        # 保证Classifier列名与performance一致
        best_oof_df = all_oof_df[
            (all_oof_df['method'] == best_method) &
            (all_oof_df['Classifier'] == best_classifier)
            ]
        best_oof_df.to_csv(os.path.join(outdir, "best_oof_predictions.csv"), encoding='utf-16', sep='\t', index=False)

        # 可选：你可以用best_oof_df的prob_1和true_label算最终的整体AUC
        final_auc = roc_auc_score(best_oof_df['true_label'], best_oof_df['prob_1'])
        print("全数据out-of-fold AUC:", final_auc)

