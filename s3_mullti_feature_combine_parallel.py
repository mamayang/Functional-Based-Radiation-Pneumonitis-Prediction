import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler


from options import parse_option
from sklearn.model_selection import train_test_split
import os
# 初始化 MinMaxScaler
import numpy as np
opt = parse_option(print_option=True)

def is_string(x):
    return isinstance(x, str)

scaler = StandardScaler()
# modality_map = {
#     1: "Clinical",
#     2: "Dose+clinical",
#     3: "GTV+clincal",
#     4: "Peri-GTV+clinical",
#     5: "Ventilation+clinical",
#     6: "Perfusion+clinical",
#
#     7: "Dose_Perfusion+clinical",
#     8: "Dose_Ventilation+clinical",
#     9: 'Dose_GTV+clinical',
#     10: 'Dose_Peri-GTV+clinical',
#
#     11: "All",
#     12: "High_Ventilation+clinical",
#     13: "High_Perfusion+clinical",
#     14: "All_High",
#
#     15: "Peri-GTV_Perfusion+clinical",
#     16: "Peri-GTV_Ventilation+clinical",
#     17: "Peri-GTV_Dose_Perfusion+clinical",
#     18: "Peri-GTV_Dose_Ventilation+clinical",
#
#     19: "Low_Ventilation+clinical",
#     20: "Low_Perfusion+clinical",
#     21: "All_Low",
#
#     22:"DVH"
#
# }
modality_map = {

    1: "HFL-R",
    2: "LFL-R",
    3: 'WL-R',

    4: "HFL-D",
    5: "LFL-D",
    6:'WL-D',

    7: "HFL-RD",
    8: "LFL-RD",

    9: "HP-RD",
    10: "HV-RD",

    11: "GTV-RD",
    12:"PTV-RD",

    13:'PTV-HV-RD',
    14:'PTV-HQ-RD',

    15:'WL-RD',
    16:'PTV-HFL-RD',
    22:"DVH"
}

modality_choose = 16
max_columns = 15000
modality= modality_map.get(modality_choose, "Unknown")

base_dir = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\Result'

ct_high_perf_path = os.path.join(base_dir, 'roi_features_ct_high_perfusion.csv')

ct_high_ven_path = os.path.join(base_dir, 'roi_features_ct_high_ventilation.csv')

ct_low_perf_path = os.path.join(base_dir, 'roi_features_ct_low_perfusion.csv')

ct_low_ven_path = os.path.join(base_dir, 'roi_features_ct_low_ventilation.csv')

ct_gtv_path = os.path.join(base_dir, 'roi_features_ct_resampled_gtv.csv')

ct_grow_path = os.path.join(base_dir, 'roi_features_ct_grow.csv')

ct_lung_path = os.path.join(base_dir, 'roi_features_ct_resampled_lung.csv')
#ON Dose map

dose_gtv_path = ct_gtv_path.replace('ct','dose')

dose_grow_path = ct_grow_path.replace('ct','dose')

dose_high_perf_path = ct_high_perf_path.replace('ct','dose')

dose_high_ven_path = ct_high_ven_path.replace('ct','dose')

dose_low_perf_path = ct_low_perf_path.replace('ct','dose')

dose_low_ven_path = ct_low_ven_path.replace('ct','dose')

dose_lung_path = ct_lung_path.replace('ct','dose')
# dose_high_perf_path = os.path.join('./dataset/for_radiomics/Dose_result', 'function' + str(opt.function_threshold),
#                                     'dose_masked_high_perfusion_dosimetric.csv')
# dose_high_ven_path = os.path.join('./dataset/for_radiomics/Dose_result', 'function' + str(opt.function_threshold), 'dose_masked_high_ventilation_dosimetric.csv')
#
# dose_low_perf_path = os.path.join('./dataset/for_radiomics/Dose_result', 'function' + str(opt.function_threshold),
#                                     'dose_masked_low_perfusion_dosimetric.csv')
# dose_low_ven_path = os.path.join('./dataset/for_radiomics/Dose_result', 'function' + str(opt.function_threshold), 'dose_masked_low_ventilation_dosimetric.csv')

ct_gtv = pd.read_csv(ct_gtv_path)
dose_gtv = pd.read_csv(dose_gtv_path)
ct_grow = pd.read_csv(ct_grow_path)
dose_grow = pd.read_csv(dose_grow_path)
dose_high_ven = pd.read_csv(dose_high_ven_path)
ct_high_ven = pd.read_csv(ct_high_ven_path)
dose_high_perf = pd.read_csv(dose_high_perf_path)
ct_low_ven = pd.read_csv(ct_low_ven_path)
ct_low_perf = pd.read_csv(ct_low_perf_path)

dose_low_ven = pd.read_csv(dose_low_ven_path)
dose_low_perf = pd.read_csv(dose_low_perf_path)

ct_high_perf = pd.read_csv(ct_high_perf_path)
ct_all_lung = pd.read_csv(ct_lung_path)
dose_all_lung = pd.read_csv(dose_lung_path)
def add_prefix_and_transpose(df, prefix):
    df.iloc[:, 0] = prefix + df.iloc[:, 0]
    return df.set_index(df.columns[0]).T


dose_gtv = add_prefix_and_transpose(dose_gtv, 'dose_gtv_')
dose_PRgtv = add_prefix_and_transpose(dose_grow, 'dose_PRgtv_')
dose_high_ven = add_prefix_and_transpose(dose_high_ven, 'dose_high_ven_')
dose_high_perf = add_prefix_and_transpose(dose_high_perf, 'dose_high_perf_')
dose_low_ven = add_prefix_and_transpose(dose_low_ven, 'dose_low_ven_')
dose_low_perf = add_prefix_and_transpose(dose_low_perf, 'dose_low_perf_')
ct_gtv = add_prefix_and_transpose(ct_gtv, 'ct_gtv_')
ct_PRgtv = add_prefix_and_transpose(ct_grow, 'ct_PRgtv_')
ct_high_ven = add_prefix_and_transpose(ct_high_ven, 'ct_high_ven_')
ct_high_perf = add_prefix_and_transpose(ct_high_perf, 'ct_high_perf_')
ct_low_ven = add_prefix_and_transpose(ct_low_ven, 'ct_low_ven_')
ct_low_perf = add_prefix_and_transpose(ct_low_perf, 'ct_low_perf_')
ct_all_lung = add_prefix_and_transpose(ct_all_lung,'ct_all_lung_')
dose_all_lung = add_prefix_and_transpose(dose_all_lung,'dose_all_lung_')

# 可更改为需要的选项
if modality == "WL-RD":
    dfs = [ct_all_lung, dose_all_lung]
elif modality == 'WL-D':
    dfs = [dose_all_lung]
elif modality == 'WL-R':
    dfs = [ct_all_lung]

elif modality == "HFL-R":
    dfs = [ct_high_ven,ct_high_perf]
elif modality == "HFL-D":
    dfs = [dose_high_ven,dose_high_perf]
elif modality == 'HFL-RD':
    dfs = [ct_high_ven,ct_high_perf,dose_high_ven,dose_high_perf]

elif modality == 'LFL-R':
    dfs = [ct_low_ven, ct_low_perf]
elif modality == 'LFL-D':
    dfs = [dose_low_ven, dose_low_perf]
elif modality == 'LFL-RD':
    dfs = [ct_low_ven, ct_low_perf, dose_low_ven, dose_low_perf]

elif modality == 'GTV-RD':
    dfs = [dose_gtv,dose_PRgtv]
elif modality == 'PTV-RD':
    dfs = [dose_PRgtv,ct_PRgtv]
elif modality == 'PTV-HV-RD':
    dfs = [dose_PRgtv,ct_PRgtv,ct_high_ven,dose_high_ven]
elif modality == 'PTV-HQ-RD':
    dfs = [dose_PRgtv,ct_PRgtv,ct_high_perf,dose_high_perf]
elif modality == 'PTV-HFL-RD':
    dfs = [dose_PRgtv,ct_PRgtv,ct_high_perf,dose_high_perf,ct_high_ven,dose_high_ven]

else:
    raise ValueError(f"Unsupported modality: {modality}")

# 按索引排序并合并 DataFrame
sorted_dfs = [df.sort_index() for df in dfs]
df_all = pd.concat(sorted_dfs, axis=1, join='outer')

# 读取临床数据并合并
df_clinical = pd.read_csv('./dataset/filter_clinical.csv')
df_all = df_all.dropna()
original_index = df_all.index
original_columns = df_all.columns

# 归一化
scaler = StandardScaler()
df_all = scaler.fit_transform(df_all)
df_all = pd.DataFrame(df_all, index=original_index, columns=original_columns)

# 处理索引
df_all.index = df_all.index.str.replace('\D', '', regex=True)
df_all.index = df_all.index.astype(int)

# 合并临床数据
df_merged = pd.merge(df_all, df_clinical, left_index=True, right_on='ID', how='inner')
df_merged = df_merged.dropna()
cols = ['ID'] + [col for col in df_merged.columns if col != 'ID']
df_merged = df_merged[cols]
df_merged = df_merged.loc[:, ~(df_merged.isna() | (df_merged == 0)).all()]

# 将数据拆分为多个文件
# 每个文件的最大列数
num_files = (df_merged.shape[1] + max_columns - 1) // max_columns

for i in range(num_files):
    start_col = i * max_columns
    end_col = min((i + 1) * max_columns, df_merged.shape[1])
    subset = df_merged.iloc[:, start_col:end_col]
    subset.to_excel(f'./Parallel/PMB/{modality}_merged_data_part_{i + 1}.xlsx', index=False)

print(f"Data has been split into {num_files} files.")

# if modality == "HFL-R":
# # 读取 CSV 文件
#     ct_gtv = pd.read_csv(ct_gtv_path)
#     ct_grow= pd.read_csv(ct_grow_path)
#     # ct_low_ven = pd.read_csv(ct_low_ven_path)
#     ct_high_ven = pd.read_csv(ct_high_ven_path)
#     # ct_low_perf = pd.read_csv(ct_low_perf_path)
#     ct_high_perf= pd.read_csv(ct_high_perf_path)
#
#     ct_gtv.iloc[:, 0] = 'gtv_' + ct_gtv.iloc[:, 0]
#     ct_grow.iloc[:, 0] = 'PRgtv_' + ct_grow.iloc[:, 0]
#     # ct_low_ven.iloc[:, 0] = 'low_ven_' + ct_low_ven.iloc[:, 0]
#     ct_high_ven.iloc[:, 0] = 'high_ven_' + ct_high_ven.iloc[:, 0]
#     # ct_low_perf.iloc[:, 0] = 'low_perf_' + ct_low_perf.iloc[:, 0]
#     ct_high_perf.iloc[:, 0] = 'high_perf_' + ct_high_perf.iloc[:, 0]
#
#     # 转置DataFrame
#     ct_gtv = ct_gtv.set_index(ct_gtv.columns[0]).T
#     ct_PRgtv = ct_grow.set_index(ct_grow.columns[0]).T
#     # ct_low_ven = ct_low_ven.set_index(ct_low_ven.columns[0]).T
#     ct_high_ven = ct_high_ven.set_index(ct_high_ven.columns[0]).T
#     # ct_low_perf = ct_low_perf.set_index(ct_low_perf.columns[0]).T
#     ct_high_perf = ct_high_perf.set_index(ct_high_perf.columns[0]).T
#
#     dfs = [ct_gtv, ct_PRgtv, ct_high_ven,ct_high_perf]
#
#     sorted_dfs = [df.sort_index() for df in dfs]
#     # 合并DataFrame
#     df_all = pd.concat(sorted_dfs, axis=1, join='outer')
#
#     df_clinical = pd.read_csv('./dataset/filter_clinical.csv')
#
#     # 将除了 'ROI_ID' 的所有列转换为浮点数
#
#
#     # df_dose.columns = ['ID'] + df_dose.columns.tolist()[1:]
#     # for col in df_dose.columns:
#     #     if col != 'Unnamed: 0':
#     #         df_dose[col] = pd.to_numeric(df_dose[col], errors='coerce')  # 将不能转换的设置为 NaN
#
#     # for col in df_clinical.columns:
#     #     if col != 'ID':
#     #         df_clinical[col] = pd.to_numeric(df_clinical[col], errors='coerce')
#     df_all = df_all.dropna()
#     original_index=df_all .index
#     original_columns=df_all.columns
#     df_all= scaler.fit_transform(df_all)
#     df_all = pd.DataFrame(df_all, index=original_index, columns=original_columns)
#     df_all.index = df_all.index.str.replace('\D', '', regex=True)
#     df_all.index = df_all.index.astype(int)
#     df_merged = pd.merge(df_all, df_clinical , left_index=True, right_on='ID', how='inner')
#     df_merged = df_merged.dropna()
#     cols = ['ID'] + [col for col in df_merged.columns if col != 'ID']
#     df_merged = df_merged[cols]
#     df_merged.to_excel('./Parallel/PMB/HFL-R_merged_data.xlsx', index=False)
#
# elif modality == "LFL-R":
# # 读取 CSV 文件
#     ct_gtv = pd.read_csv(ct_gtv_path)
#     ct_grow= pd.read_csv(ct_grow_path)
#     ct_low_ven = pd.read_csv(ct_low_ven_path)
#     # ct_high_ven = pd.read_csv(ct_high_ven_path)
#     ct_low_perf = pd.read_csv(ct_low_perf_path)
#     # ct_high_perf= pd.read_csv(ct_high_perf_path)
#
#     ct_gtv.iloc[:, 0] = 'gtv_' + ct_gtv.iloc[:, 0]
#     ct_grow.iloc[:, 0] = 'PRgtv_' + ct_grow.iloc[:, 0]
#     ct_low_ven.iloc[:, 0] = 'low_ven_' + ct_low_ven.iloc[:, 0]
#     # ct_high_ven.iloc[:, 0] = 'high_ven_' + ct_high_ven.iloc[:, 0]
#     ct_low_perf.iloc[:, 0] = 'low_perf_' + ct_low_perf.iloc[:, 0]
#     # ct_high_perf.iloc[:, 0] = 'high_perf_' + ct_high_perf.iloc[:, 0]
#     #
#     # 转置DataFrame
#     ct_gtv = ct_gtv.set_index(ct_gtv.columns[0]).T
#     ct_PRgtv = ct_grow.set_index(ct_grow.columns[0]).T
#     ct_low_ven = ct_low_ven.set_index(ct_low_ven.columns[0]).T
#     # ct_high_ven = ct_high_ven.set_index(ct_high_ven.columns[0]).T
#     ct_low_perf = ct_low_perf.set_index(ct_low_perf.columns[0]).T
#     # ct_high_perf = ct_high_perf.set_index(ct_high_perf.columns[0]).T
#
#     dfs = [ct_gtv, ct_PRgtv, ct_low_ven,ct_low_perf]
#
#     sorted_dfs = [df.sort_index() for df in dfs]
#     # 合并DataFrame
#     df_all = pd.concat(sorted_dfs, axis=1, join='outer')
#
#     df_clinical = pd.read_csv('./dataset/filter_clinical.csv')
#
#     # 将除了 'ROI_ID' 的所有列转换为浮点数
#
#
#     # df_dose.columns = ['ID'] + df_dose.columns.tolist()[1:]
#     # for col in df_dose.columns:
#     #     if col != 'Unnamed: 0':
#     #         df_dose[col] = pd.to_numeric(df_dose[col], errors='coerce')  # 将不能转换的设置为 NaN
#
#     # for col in df_clinical.columns:
#     #     if col != 'ID':
#     #         df_clinical[col] = pd.to_numeric(df_clinical[col], errors='coerce')
#     df_all = df_all.dropna()
#     original_index=df_all .index
#     original_columns=df_all.columns
#     df_all= scaler.fit_transform(df_all)
#     df_all = pd.DataFrame(df_all, index=original_index, columns=original_columns)
#     df_all.index = df_all.index.str.replace('\D', '', regex=True)
#     df_all.index = df_all.index.astype(int)
#     df_merged = pd.merge(df_all, df_clinical , left_index=True, right_on='ID', how='inner')
#     df_merged = df_merged.dropna()
#     cols = ['ID'] + [col for col in df_merged.columns if col != 'ID']
#     df_merged = df_merged[cols]
#     df_merged.to_excel('./Parallel/PMB/LFL-R_merged_data.xlsx', index=False)
#
# elif modality == "LFL-D":
# # 读取 CSV 文件
#     dose_gtv = pd.read_csv(dose_gtv_path)
#     dose_grow= pd.read_csv(dose_grow_path)
#     dose_low_ven = pd.read_csv(dose_low_ven_path)
#     # ct_high_ven = pd.read_csv(ct_high_ven_path)
#     dose_low_perf = pd.read_csv(dose_low_perf_path)
#     # ct_high_perf= pd.read_csv(ct_high_perf_path)
#
#     dose_gtv.iloc[:, 0] = 'gtv_' + dose_gtv.iloc[:, 0]
#     dose_grow.iloc[:, 0] = 'PRgtv_' + dose_grow.iloc[:, 0]
#     dose_low_ven.iloc[:, 0] = 'low_ven_' + dose_low_ven.iloc[:, 0]
#     # ct_high_ven.iloc[:, 0] = 'high_ven_' + ct_high_ven.iloc[:, 0]
#     dose_low_perf.iloc[:, 0] = 'low_perf_' + dose_low_perf.iloc[:, 0]
#     # ct_high_perf.iloc[:, 0] = 'high_perf_' + ct_high_perf.iloc[:, 0]
#     #
#     # 转置DataFrame
#     dose_gtv = dose_gtv.set_index(dose_gtv.columns[0]).T
#     dose_PRgtv = dose_grow.set_index(dose_grow.columns[0]).T
#     dose_low_ven = dose_low_ven.set_index(dose_low_ven.columns[0]).T
#     # ct_high_ven = ct_high_ven.set_index(ct_high_ven.columns[0]).T
#     dose_low_perf = dose_low_perf.set_index(dose_low_perf.columns[0]).T
#     # ct_high_perf = ct_high_perf.set_index(ct_high_perf.columns[0]).T
#
#     dfs = [dose_gtv, dose_PRgtv, dose_low_ven,dose_low_perf]
#
#     sorted_dfs = [df.sort_index() for df in dfs]
#     # 合并DataFrame
#     df_all = pd.concat(sorted_dfs, axis=1, join='outer')
#
#     df_clinical = pd.read_csv('./dataset/filter_clinical.csv')
#
#     # 将除了 'ROI_ID' 的所有列转换为浮点数
#
#
#     # df_dose.columns = ['ID'] + df_dose.columns.tolist()[1:]
#     # for col in df_dose.columns:
#     #     if col != 'Unnamed: 0':
#     #         df_dose[col] = pd.to_numeric(df_dose[col], errors='coerce')  # 将不能转换的设置为 NaN
#
#     # for col in df_clinical.columns:
#     #     if col != 'ID':
#     #         df_clinical[col] = pd.to_numeric(df_clinical[col], errors='coerce')
#     df_all = df_all.dropna()
#     original_index=df_all .index
#     original_columns=df_all.columns
#     df_all= scaler.fit_transform(df_all)
#     df_all = pd.DataFrame(df_all, index=original_index, columns=original_columns)
#     df_all.index = df_all.index.str.replace('\D', '', regex=True)
#     df_all.index = df_all.index.astype(int)
#     df_merged = pd.merge(df_all, df_clinical , left_index=True, right_on='ID', how='inner')
#     df_merged = df_merged.dropna()
#     cols = ['ID'] + [col for col in df_merged.columns if col != 'ID']
#     df_merged = df_merged[cols]
#
#     df_merged.to_excel('./Parallel/PMB/LFL-D_merged_data.xlsx', index=False)
#
# # elif modality == "HFL-D":
# # # 读取 CSV 文件
# #     dose_gtv = pd.read_csv(dose_gtv_path)
# #     dose_grow= pd.read_csv(dose_grow_path)
# #     dose_high_ven = pd.read_csv(dose_high_ven_path)
# #     # ct_high_ven = pd.read_csv(ct_high_ven_path)
# #     dose_high_perf = pd.read_csv(dose_high_perf_path)
# #     # ct_high_perf= pd.read_csv(ct_high_perf_path)
# #
# #     dose_gtv.iloc[:, 0] = 'gtv_' + dose_gtv.iloc[:, 0]
# #     dose_grow.iloc[:, 0] = 'PRgtv_' + dose_grow.iloc[:, 0]
# #     dose_high_ven.iloc[:, 0] = 'low_ven_' + dose_high_ven.iloc[:, 0]
# #     # ct_high_ven.iloc[:, 0] = 'high_ven_' + ct_high_ven.iloc[:, 0]
# #     dose_high_perf.iloc[:, 0] = 'low_perf_' + dose_high_perf.iloc[:, 0]
# #     # ct_high_perf.iloc[:, 0] = 'high_perf_' + ct_high_perf.iloc[:, 0]
# #     #
# #     # 转置DataFrame
# #     dose_gtv = dose_gtv.set_index(dose_gtv.columns[0]).T
# #     dose_PRgtv = dose_grow.set_index(dose_grow.columns[0]).T
# #     dose_high_ven = dose_high_ven.set_index(dose_high_ven.columns[0]).T
# #     # ct_high_ven = ct_high_ven.set_index(ct_high_ven.columns[0]).T
# #     dose_high_perf = dose_high_perf.set_index(dose_high_perf.columns[0]).T
# #     # ct_high_perf = ct_high_perf.set_index(ct_high_perf.columns[0]).T
# #
# #     dfs = [dose_gtv, dose_PRgtv, dose_high_ven,dose_high_perf]
# #
# #     sorted_dfs = [df.sort_index() for df in dfs]
# #     # 合并DataFrame
# #     df_all = pd.concat(sorted_dfs, axis=1, join='outer')
# #
# #     df_clinical = pd.read_csv('./dataset/filter_clinical.csv')
# #
# #     # 将除了 'ROI_ID' 的所有列转换为浮点数
# #
# #
# #     # df_dose.columns = ['ID'] + df_dose.columns.tolist()[1:]
# #     # for col in df_dose.columns:
# #     #     if col != 'Unnamed: 0':
# #     #         df_dose[col] = pd.to_numeric(df_dose[col], errors='coerce')  # 将不能转换的设置为 NaN
# #
# #     # for col in df_clinical.columns:
# #     #     if col != 'ID':
# #     #         df_clinical[col] = pd.to_numeric(df_clinical[col], errors='coerce')
# #     df_all = df_all.dropna()
# #     original_index=df_all .index
# #     original_columns=df_all.columns
# #     df_all= scaler.fit_transform(df_all)
# #     df_all = pd.DataFrame(df_all, index=original_index, columns=original_columns)
# #     df_all.index = df_all.index.str.replace('\D', '', regex=True)
# #     df_all.index = df_all.index.astype(int)
# #     df_merged = pd.merge(df_all, df_clinical , left_index=True, right_on='ID', how='inner')
# #     df_merged = df_merged.dropna()
# #     cols = ['ID'] + [col for col in df_merged.columns if col != 'ID']
# #     df_merged = df_merged[cols]
# #     df_merged.to_excel('./Parallel/PMB/HFL-D_merged_data.xlsx', index=False)
#
# elif modality == "LFL-RD":
# # 读取 CSV 文件
#     ct_gtv = pd.read_csv(ct_gtv_path)
#     dose_gtv = pd.read_csv(dose_gtv_path)
#
#     ct_grow= pd.read_csv(ct_grow_path)
#     dose_grow = pd.read_csv(dose_grow_path)
#
#     ct_low_ven = pd.read_csv(ct_low_ven_path)
#     dose_low_ven = pd.read_csv(dose_low_ven_path)
#     # ct_high_ven = pd.read_csv(ct_high_ven_path)
#
#     ct_low_perf = pd.read_csv(ct_low_perf_path)
#     dose_low_perf = pd.read_csv(dose_low_perf_path)
#     # ct_high_perf= pd.read_csv(ct_high_perf_path)
#
#     ct_gtv.iloc[:, 0] = 'ct_gtv_' + ct_gtv.iloc[:, 0]
#     ct_grow.iloc[:, 0] = 'ct_PRgtv_' + ct_grow.iloc[:, 0]
#     ct_low_ven.iloc[:, 0] = 'ct_low_ven_' + ct_low_ven.iloc[:, 0]
#     ct_low_perf.iloc[:, 0] = 'ct_low_perf_' + ct_low_perf.iloc[:, 0]
#     # ct_high_perf.iloc[:, 0] = 'high_perf_' + ct_high_perf.iloc[:, 0]
#
#     dose_gtv.iloc[:, 0] = 'dose_gtv_' + dose_gtv.iloc[:, 0]
#     dose_grow.iloc[:, 0] = 'dose_PRgtv_' + dose_grow.iloc[:, 0]
#     dose_low_ven.iloc[:, 0] = 'dose_low_ven_' + dose_low_ven.iloc[:, 0]
#     # ct_high_ven.iloc[:, 0] = 'high_ven_' + ct_high_ven.iloc[:, 0]
#     dose_low_perf.iloc[:, 0] = 'dose_low_perf_' + dose_low_perf.iloc[:, 0]
#
#     # 转置DataFrame
#     ct_gtv = ct_gtv.set_index(ct_gtv.columns[0]).T
#     ct_PRgtv = ct_grow.set_index(ct_grow.columns[0]).T
#     ct_low_ven = ct_low_ven.set_index(ct_low_ven.columns[0]).T
#     # ct_high_ven = ct_high_ven.set_index(ct_high_ven.columns[0]).T
#     ct_low_perf = ct_low_perf.set_index(ct_low_perf.columns[0]).T
#     # ct_high_perf = ct_high_perf.set_index(ct_high_perf.columns[0]).T
#
#     dose_gtv = dose_gtv.set_index(dose_gtv.columns[0]).T
#     dose_PRgtv = dose_grow.set_index(dose_grow.columns[0]).T
#     dose_low_ven = dose_low_ven.set_index(dose_low_ven.columns[0]).T
#     # ct_high_ven = ct_high_ven.set_index(ct_high_ven.columns[0]).T
#     dose_low_perf = dose_low_perf.set_index(dose_low_perf.columns[0]).T
#
#
#     # dfs = [ct_gtv, ct_PRgtv, ct_low_ven,ct_low_perf, dose_low_ven,dose_low_perf]
#     dfs = [ct_gtv, ct_PRgtv, dose_gtv, dose_PRgtv, ct_low_ven,ct_low_perf, dose_low_ven,dose_low_perf]
#     sorted_dfs = [df.sort_index() for df in dfs]
#     # 合并DataFrame
#     df_all = pd.concat(sorted_dfs, axis=1, join='outer')
#
#     df_clinical = pd.read_csv('./dataset/filter_clinical.csv')
#
#     # 将除了 'ROI_ID' 的所有列转换为浮点数
#
#
#     # df_dose.columns = ['ID'] + df_dose.columns.tolist()[1:]
#     # for col in df_dose.columns:
#     #     if col != 'Unnamed: 0':
#     #         df_dose[col] = pd.to_numeric(df_dose[col], errors='coerce')  # 将不能转换的设置为 NaN
#
#     # for col in df_clinical.columns:
#     #     if col != 'ID':
#     #         df_clinical[col] = pd.to_numeric(df_clinical[col], errors='coerce')
#     df_all = df_all.dropna()
#     original_index=df_all .index
#     original_columns=df_all.columns
#     df_all= scaler.fit_transform(df_all)
#     df_all = pd.DataFrame(df_all, index=original_index, columns=original_columns)
#     df_all.index = df_all.index.str.replace('\D', '', regex=True)
#     df_all.index = df_all.index.astype(int)
#     df_merged = pd.merge(df_all, df_clinical , left_index=True, right_on='ID', how='inner')
#     df_merged = df_merged.dropna()
#     cols = ['ID'] + [col for col in df_merged.columns if col != 'ID']
#     df_merged = df_merged[cols]
#
#     df_merged = df_merged.loc[:, ~(df_merged.isna() | (df_merged == 0)).all()]
#       # Excel 的最大列数限制
#
#     # 计算需要多少个文件
#     num_files = (df_merged.shape[1] + max_columns - 1) // max_columns
#
#     for i in range(num_files):
#         start_col = i * max_columns
#         end_col = min((i + 1) * max_columns, df_merged.shape[1])
#         subset = df_merged.iloc[:, start_col:end_col]
#         subset.to_excel(f'./Parallel/PMB/LFL-RD_merged_data_part_{i + 1}.xlsx', index=False)
#
#     print(f"Data has been split into {num_files} files.")
#
# # elif modality == "HFL-RD":
# # # 读取 CSV 文件
# #     ct_gtv = pd.read_csv(ct_gtv_path)
# #     dose_gtv = pd.read_csv(dose_gtv_path)
# #     ct_grow= pd.read_csv(ct_grow_path)
# #     dose_grow = pd.read_csv(dose_grow_path)
# #
# #     dose_high_ven = pd.read_csv(dose_high_ven_path)
# #     # ct_high_ven = pd.read_csv(ct_high_ven_path)
# #     dose_high_perf = pd.read_csv(dose_high_perf_path)
# #     # ct_high_perf= pd.read_csv(ct_high_perf_path)
# #
# #     dose_gtv.iloc[:, 0] = 'dose_gtv_' + dose_gtv.iloc[:, 0]
# #     dose_grow.iloc[:, 0] = 'dose_PRgtv_' + dose_grow.iloc[:, 0]
# #     dose_high_ven.iloc[:, 0] = 'dose_high_ven_' + dose_high_ven.iloc[:, 0]
# #     # ct_high_ven.iloc[:, 0] = 'high_ven_' + ct_high_ven.iloc[:, 0]
# #     dose_high_perf.iloc[:, 0] = 'dose_high_perf_' + dose_high_perf.iloc[:, 0]
# #     # ct_high_perf.iloc[:, 0] = 'high_perf_' + ct_high_perf.iloc[:, 0]
# #     #
# #     # 转置DataFrame
# #     dose_gtv = dose_gtv.set_index(dose_gtv.columns[0]).T
# #     dose_PRgtv = dose_grow.set_index(dose_grow.columns[0]).T
# #     dose_high_ven = dose_high_ven.set_index(dose_high_ven.columns[0]).T
# #     # ct_high_ven = ct_high_ven.set_index(ct_high_ven.columns[0]).T
# #     dose_high_perf = dose_high_perf.set_index(dose_high_perf.columns[0]).T
# #
# #     ct_high_ven = pd.read_csv(ct_high_ven_path)
# #     # ct_low_perf = pd.read_csv(ct_low_perf_path)
# #     ct_high_perf= pd.read_csv(ct_high_perf_path)
# #
# #     ct_gtv.iloc[:, 0] = 'ct_gtv_' + ct_gtv.iloc[:, 0]
# #     ct_grow.iloc[:, 0] = 'ct_PRgtv_' + ct_grow.iloc[:, 0]
# #     # ct_low_ven.iloc[:, 0] = 'low_ven_' + ct_low_ven.iloc[:, 0]
# #     ct_high_ven.iloc[:, 0] = 'ct_high_ven_' + ct_high_ven.iloc[:, 0]
# #     # ct_low_perf.iloc[:, 0] = 'low_perf_' + ct_low_perf.iloc[:, 0]
# #     ct_high_perf.iloc[:, 0] = 'ct_high_perf_' + ct_high_perf.iloc[:, 0]
# #
# #     # 转置DataFrame
# #     ct_gtv = ct_gtv.set_index(ct_gtv.columns[0]).T
# #     ct_PRgtv = ct_grow.set_index(ct_grow.columns[0]).T
# #     # ct_low_ven = ct_low_ven.set_index(ct_low_ven.columns[0]).T
# #     ct_high_ven = ct_high_ven.set_index(ct_high_ven.columns[0]).T
# #     # ct_low_perf = ct_low_perf.set_index(ct_low_perf.columns[0]).T
# #     ct_high_perf = ct_high_perf.set_index(ct_high_perf.columns[0]).T
# #
# #
# #     # dfs = [ct_gtv, ct_PRgtv, ct_high_ven,ct_high_perf,dose_high_ven,dose_high_perf]
# #     dfs = [ct_gtv, ct_PRgtv,dose_gtv, dose_PRgtv, ct_high_ven,ct_high_perf,dose_high_ven,dose_high_perf]
# #     sorted_dfs = [df.sort_index() for df in dfs]
# #     # 合并DataFrame
# #     df_all = pd.concat(sorted_dfs, axis=1, join='outer')
# #
# #     df_clinical = pd.read_csv('./dataset/filter_clinical.csv')
# #
# #     # 将除了 'ROI_ID' 的所有列转换为浮点数
# #
# #
# #     # df_dose.columns = ['ID'] + df_dose.columns.tolist()[1:]
# #     # for col in df_dose.columns:
# #     #     if col != 'Unnamed: 0':
# #     #         df_dose[col] = pd.to_numeric(df_dose[col], errors='coerce')  # 将不能转换的设置为 NaN
# #
# #     # for col in df_clinical.columns:
# #     #     if col != 'ID':
# #     #         df_clinical[col] = pd.to_numeric(df_clinical[col], errors='coerce')
# #     df_all = df_all.dropna()
# #     original_index=df_all .index
# #     original_columns=df_all.columns
# #     df_all= scaler.fit_transform(df_all)
# #     df_all = pd.DataFrame(df_all, index=original_index, columns=original_columns)
# #     df_all.index = df_all.index.str.replace('\D', '', regex=True)
# #     df_all.index = df_all.index.astype(int)
# #     df_merged = pd.merge(df_all, df_clinical, left_index=True, right_on='ID', how='inner')
# #     df_merged = df_merged.dropna()
# #     cols = ['ID'] + [col for col in df_merged.columns if col != 'ID']
# #     df_merged = df_merged[cols]
# #
# #     df_merged = df_merged.loc[:, ~(df_merged.isna() | (df_merged == 0)).all()]
# #
# #
# #     # 计算需要多少个文件
# #     num_files = (df_merged.shape[1] + max_columns - 1) // max_columns
# #
# #     for i in range(num_files):
# #         start_col = i * max_columns
# #         end_col = min((i + 1) * max_columns, df_merged.shape[1])
# #         subset = df_merged.iloc[:, start_col:end_col]
# #         subset.to_excel(f'./Parallel/PMB/HFL-RD_merged_data_part_{i + 1}.xlsx', index=False)


