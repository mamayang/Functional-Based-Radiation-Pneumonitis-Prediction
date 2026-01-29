import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif,SelectKBest
from utils import load_data
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
from collections import Counter
from utils.feature_selection import LassoSelection, FscoreSelection, MISelection
from options import parse_option
from collections import defaultdict
from sklearn.feature_selection import VarianceThreshold
import glob
# Load your data
# Assuming `data` as DataFrame with last column as labels
# Replace this with actual data loading code

opt = parse_option(print_option=True)

def feature_selection_fun(train_data = None, n_iter=100, sample_ratio=0.7):

    if train_data is None :
        raise ValueError("the sifted columns cannot be None.")
    else:
        features = train_data.iloc[:, :-1]

        # Determine the minimum number of features to retain
        selector = VarianceThreshold(threshold=0)
        selector.fit(features)

        # Get mask of features with zero variance
        constant_mask = selector.variances_ == 0

        # Get feature names
        constant_features = features.columns[constant_mask]

        # 1. Get mask of non-constant features
        non_constant_mask = ~constant_mask  # Invert to obtain non-constant feature mask

        # 2. Get names of non-constant features
        feature_names = features.columns[non_constant_mask].tolist()

        # 3. Update feature data
        features = features[feature_names]
        print("Features with zero variance:")
        print(constant_features)
        print("\nTotal of {} features with zero variance".format(len(constant_features)))


        target = train_data.iloc[:, -1]
        selected_features = set()
        all_selected_features = []
        dataset_dict_fs = {'LASSO': [],
                           'Fscore': [],
                           'MI': []}
        
        # 100 times random down-sampling (70% of patients each time)
        for i in range(n_iter):
            # Randomly sample 70% of patients
            n_samples = int(len(train_data) * sample_ratio)
            sampled_indices = train_data.sample(n=n_samples, random_state=None).index
            sampled_features = features.loc[sampled_indices]
            sampled_target = target.loc[sampled_indices]
            
            print(f"Iteration {i + 1}: sampled {len(sampled_indices)} patients (70% of total)")
            print("Feature Selection...", i)
            
            # Remove constant features (zero variance) before ANOVA
            variance_selector = VarianceThreshold(threshold=0)
            variance_selector.fit(sampled_features)
            non_constant_mask = variance_selector.get_support()
            
            # Identify constant features
            constant_features = sampled_features.columns[~non_constant_mask].tolist()
            if len(constant_features) > 0:
                print(f"  Constant features removed before ANOVA ({len(constant_features)}): {constant_features[:10]}{'...' if len(constant_features) > 10 else ''}")
            
            # Use only non-constant features
            sampled_features = sampled_features.loc[:, non_constant_mask]
            
           
            f_values, p_values = f_classif(sampled_features, sampled_target)
            
            # Keep significant features with p < alpha (ANOVA > 0 understood as statistically significant, i.e., p < alpha)
            anova_features_mask = p_values < opt.alpha
            anova_selected_features = sampled_features.columns[anova_features_mask].tolist()
            anova_features_df = sampled_features[anova_selected_features]
            
            print(f"  After ANOVA: {len(anova_selected_features)} features (p < {opt.alpha})")
            
            # Check if there are still features after ANOVA filtering
            if len(anova_selected_features) == 0:
                print(f"  Warning: No features passed ANOVA filter (p < {opt.alpha}), skipping this iteration")
                # 添加空列表以保持数据结构一致
                dataset_dict_fs['LASSO'].append([])
                dataset_dict_fs['Fscore'].append([])
                dataset_dict_fs['MI'].append([])
                continue
            
            # Step 2: Supervised feature selection (Lasso/MI/F-score) on ANOVA-filtered features
            selected_lasso_features = LassoSelection(anova_features_df, sampled_target, opt)
            dataset_dict_fs['LASSO'].append(selected_lasso_features)
            
            selected_fscore_features = FscoreSelection(anova_features_df, sampled_target, opt)
            dataset_dict_fs['Fscore'].append(selected_fscore_features)
            
            selected_MI_features = MISelection(anova_features_df, sampled_target, opt)
            dataset_dict_fs['MI'].append(selected_MI_features)




            # 更新选定特征集
        lasso_feature_counts = defaultdict(int)
        for features_list in dataset_dict_fs['LASSO']:
            for feature in features_list:
                lasso_feature_counts[feature] += 1

        lasso_sorted_features = sorted( lasso_feature_counts, key= lasso_feature_counts.get, reverse=True)

        lasso_selected_features = features[lasso_sorted_features]
        lasso_correlation_matrix = lasso_selected_features.corr()

        lasso_to_drop = set()
        for i in range(len(lasso_sorted_features)):
            for j in range(i + 1, len(lasso_sorted_features)):
                if abs(lasso_correlation_matrix.iloc[i, j]) > opt.corr_threshold:
                    # Choose feature to drop (here simply the later one in the list)
                    lasso_to_drop.add(lasso_sorted_features[j])
        lasso_reduced = lasso_selected_features.drop(columns=lasso_to_drop)

        fscore_feature_counts = defaultdict(int)
        for features_list in dataset_dict_fs['Fscore']:
            for feature in features_list:
                fscore_feature_counts[feature] += 1

        fscore_sorted_features = sorted(fscore_feature_counts, key=fscore_feature_counts.get, reverse=True)

        fscore_selected_features = features[fscore_sorted_features]
        fscore_correlation_matrix = fscore_selected_features.corr()

        fscore_to_drop = set()
        for i in range(len(fscore_sorted_features)):
            for j in range(i + 1, len(fscore_sorted_features)):
                if abs(fscore_correlation_matrix.iloc[i, j]) > opt.corr_threshold:
                    # Choose feature to drop (here simply the later one in the list)
                    fscore_to_drop.add(fscore_sorted_features[j])
        fscore_reduced = fscore_selected_features.drop(columns=fscore_to_drop)
        
        mi_feature_counts = defaultdict(int)
        for features_list in dataset_dict_fs['MI']:
            for feature in features_list:
                mi_feature_counts[feature] += 1

        mi_sorted_features = sorted(mi_feature_counts, key=mi_feature_counts.get, reverse=True)

        mi_selected_features = features[mi_sorted_features]
        mi_correlation_matrix = mi_selected_features.corr()

        mi_to_drop = set()
        for i in range(len(mi_sorted_features)):
            for j in range(i + 1, len(mi_sorted_features)):
                if abs(mi_correlation_matrix.iloc[i, j]) > opt.corr_threshold:
                    # Choose feature to drop (here simply the later one in the list)
                    mi_to_drop.add(mi_sorted_features[j])
        mi_reduced = mi_selected_features.drop(columns=mi_to_drop)



    return list(lasso_reduced.columns),list(fscore_reduced.columns),list(mi_reduced.columns)

modalitylist = ["LFL-D","LFL-RD","PTV-HQ-RD","GTV-RD",
                'WL-D','WL-R','WL-RD','HFL-R', "HFL-D"]

n_splits = 10

for modality in modalitylist:
    opt.modality = modality
    possible_filenames = [
        f"{modality}_merged_data_part_1_normalized.csv",
        f"{modality}_merged_data_normalized.csv"
    ]
    file_list = []
    for filename in possible_filenames:
        potential_path = os.path.join('./Parallel/PMB/', filename)
        if os.path.exists(potential_path):
            file_list.append(potential_path)
            break

    label_col = 'Grade'
    opt.data_path = file_list
    opt.label_col = label_col
    data = load_data(file_list)
    additional_columns = list(data.columns[-10:-1].values)
    exclude_cols = additional_columns.copy()
    exclude_cols.insert(0, 'ID')
    opt.exclude_cols = exclude_cols
    additional_columns.append('Grade')
    data_df = data.drop(columns=opt.exclude_cols)
    data_df = data_df.drop(data_df.columns[data_df.eq(0).all()], axis=1)

    dtype_dict = {}
    for column in data_df.columns:
        try:
            if data_df[column].nunique() == 1:
                continue
            if data_df[column].dtype == np.int64:
                dtype_dict[column] = np.int8
        except Exception as e:
            print(f"Error occurred while converting column '{column}': {e}")
    data_df = data_df.astype(dtype_dict)

    # Extract label and ID
    labels = data[label_col].values
    ids = data['ID'].values if 'ID' in data.columns else np.arange(len(data_df))
    # 10-fold StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_dir = os.path.join('./dataset/Dataset_split/PMB', 'Parallel', f'{n_splits}-fold', opt.modality,
                            f'f_ratio_{opt.feature_ratio}_cor_{opt.corr_threshold}')
    os.makedirs(test_dir, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_df, labels)):
        print(f'===> Fold {fold + 1}/{n_splits}')
        train_df = data_df.iloc[train_idx].copy()
        test_df = data_df.iloc[test_idx].copy()
        train_label = labels[train_idx]
        test_label = labels[test_idx]

        train_df[label_col] = train_label  # Ensure the label column exists

        # Feature selection
        lasso_selected_features, fscore_selected_features, mi_selected_features = feature_selection_fun(
            train_data=train_df)
        print(f"Number of LASSO features: {len(lasso_selected_features)}")
        print(f"Number of Fscore features: {len(fscore_selected_features)}")
        print(f"Number of MI features: {len(mi_selected_features)}")
        # Save IDs/filenames of the train set of each fold to txt
        # Assume you have 'ID' or 'filename' column, otherwise save index
        if 'ID' in data.columns:
            train_ids = data.iloc[train_idx]['ID'].tolist()
        elif 'filename' in data.columns:
            train_ids = data.iloc[train_idx]['filename'].tolist()
        else:
            train_ids = list(train_idx)
        train_txt_path = os.path.join(test_dir, f'fold_{fold + 1}_train.txt')
        with open(train_txt_path, 'w') as f:
            for tid in train_ids:
                f.write(str(tid) + '\n')
        print(f'Fold {fold + 1} train file saved: {train_txt_path}')

        # Save IDs/filenames of the test set of each fold to txt
        if 'ID' in data.columns:
            test_ids = data.iloc[test_idx]['ID'].tolist()
        elif 'filename' in data.columns:
            test_ids = data.iloc[test_idx]['filename'].tolist()
        else:
            test_ids = list(test_idx)
        test_txt_path = os.path.join(test_dir, f'fold_{fold + 1}_test.txt')
        with open(test_txt_path, 'w') as f:
            for tid in test_ids:
                f.write(str(tid) + '\n')
        print(f'Fold {fold + 1} test file saved: {test_txt_path}')

        # Save feature selection results
        lasso_to_select = (['ID'] if 'ID' in train_df.columns else []) + lasso_selected_features + additional_columns
        fscore_to_select = (['ID'] if 'ID' in train_df.columns else []) + fscore_selected_features + additional_columns
        mi_to_select = (['ID'] if 'ID' in train_df.columns else []) + mi_selected_features + additional_columns

        pd.DataFrame(lasso_to_select).to_csv(
            os.path.join(test_dir, f'fold_{fold + 1}_selected_lasso_features_with_info.csv'), index=False)
        pd.DataFrame(fscore_to_select).to_csv(
            os.path.join(test_dir, f'fold_{fold + 1}_selected_fscore_features_with_info.csv'), index=False)
        pd.DataFrame(mi_to_select).to_csv(os.path.join(test_dir, f'fold_{fold + 1}_selected_mi_features_with_info.csv'),
                                          index=False)
    print(f"All test files of all folds have been saved to: {test_dir}")
