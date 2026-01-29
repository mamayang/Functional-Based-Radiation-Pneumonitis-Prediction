import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV
import numpy as np

def LassoSelection(rus_feature_df, rus_target_df, opt):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV
    
    # 检查特征数量
    if rus_feature_df.shape[1] == 0:
        print(">>> Warning: No features available after filtering, returning empty list")
        return []
    
    # 准备数据（保持原有格式）
    x_train = rus_feature_df.to_numpy()
    y_train = rus_target_df.to_numpy()
    
    # 使用Pipeline包含标准化和Lasso模型
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', Lasso(max_iter=5000, tol=1e-4))  # 增加迭代次数，降低容差
    ])
    
    # 使用GridSearchCV搜索最优alpha（采用新方法的搜索范围）
    search = GridSearchCV(
        pipeline, 
        {'model__alpha': np.arange(0.1, 10, 0.1)}, 
        cv=5, 
        scoring="neg_mean_squared_error",
        verbose=0
    )
    
    try:
        search.fit(x_train, y_train)
        # 检查是否有有效的拟合结果
        if np.isnan(search.best_score_):
            print(">>> Warning: All CV folds failed in LassoSelection, returning empty list")
            return []
    except Exception as e:
        print(f">>> Error in LassoSelection: {e}")
        print(f">>> Feature shape: {x_train.shape}, Target shape: {y_train.shape}")
        return []
    
    # 获取最优模型的系数
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    
    # 选择系数非零的特征（新方法的核心逻辑）
    selected_indices = importance > 0
    
    # 如果选中的特征数量超过限制，按重要性排序选择前max_feat个
    n_feats = rus_feature_df.shape[1]
    if n_feats < 30:
        max_feat = n_feats
    else:
        max_feat = int(opt.feature_ratio * n_feats)
        if max_feat > 30:
            max_feat = 30
    
    # 如果选中的特征超过限制，按重要性排序
    if np.sum(selected_indices) > max_feat:
        # 按重要性排序，选择前max_feat个
        sorted_indices = np.argsort(importance)[::-1]
        selected_indices = np.zeros_like(selected_indices, dtype=bool)
        selected_indices[sorted_indices[:max_feat]] = True
    
    # 获取选中的特征名称
    selected_feats_lasso = rus_feature_df.columns[selected_indices].to_list()
    
    print(">>> # of Lasso Selected Features : %d" % len(selected_feats_lasso))
    
    return selected_feats_lasso


def FscoreSelection(rus_feature_df, rus_target_df, opt):
    # train_X = train_df.drop(opt.label_col, axis=1)
    # train_y = train_df[opt.label_col]
    n_feats = rus_feature_df.shape[1]  # Number of features
    if n_feats < 30:
        max_feat = n_feats
    else:
        max_feat = int(opt.feature_ratio * n_feats)
        if max_feat > 30:
            max_feat = 30
    f_selector = SelectKBest(f_classif, k=max_feat)
    f_selector.fit(rus_feature_df, rus_target_df)

    selected_feats_fscore = rus_feature_df.columns[f_selector.get_support()].to_list()
    print(">>> # of F-score Selected Features : %d" % len(selected_feats_fscore))
    # pd.DataFrame(selected_feats).to_excel(file_name, index=False, header=False)
    return selected_feats_fscore


def MISelection(rus_feature_df, rus_target_df, opt):
    # train_X = train_df.drop(opt.label_col, axis=1)
    # train_y = train_df[opt.label_col]
    n_feats = rus_feature_df.shape[1]  # Number of features
    if n_feats < 30:
        max_feat = n_feats
    else:
        max_feat = int(opt.feature_ratio * n_feats)
        if max_feat > 30:
            max_feat = 30
    mi_selector = SelectKBest(mutual_info_classif, k=max_feat)
    mi_selector.fit(rus_feature_df, rus_target_df)

    selected_feats_mi =rus_feature_df.columns[mi_selector.get_support()].to_list()
    print(">>> # of Mutual Information Selected Features : %d" % len(selected_feats_mi))
    #
    # train_df_mi = train_df[[opt.label_col]+selected_feats]
    # valid_df_mi = valid_df[[opt.label_col]+selected_feats]
    #
    # # Save selected features list as excel file
    # file_name = os.path.join(opt.exp, 'MI_selected_features.xlsx')
    # pd.DataFrame(selected_feats).to_excel(file_name, index=False, header=False)

    return selected_feats_mi