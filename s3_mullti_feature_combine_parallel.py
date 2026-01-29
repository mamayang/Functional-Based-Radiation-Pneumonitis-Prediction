import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler


from options import parse_option
from sklearn.model_selection import train_test_split
import os

import numpy as np
opt = parse_option(print_option=True)

def is_string(x):
    return isinstance(x, str)

scaler = StandardScaler()

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

# Feature region selection settings
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

# Sort by index and merge to the DataFrames
sorted_dfs = [df.sort_index() for df in dfs]
df_all = pd.concat(sorted_dfs, axis=1, join='outer')

# Get clinical information
df_clinical = pd.read_csv('./dataset/filter_clinical.csv')
df_all = df_all.dropna()
original_index = df_all.index
original_columns = df_all.columns

# Standardization
scaler = StandardScaler()
df_all = scaler.fit_transform(df_all)
df_all = pd.DataFrame(df_all, index=original_index, columns=original_columns)

# Remove all non-numeric characters from the index.
df_all.index = df_all.index.str.replace('\D', '', regex=True)
df_all.index = df_all.index.astype(int)

# Combine the clinical information
df_merged = pd.merge(df_all, df_clinical, left_index=True, right_on='ID', how='inner')
df_merged = df_merged.dropna()
cols = ['ID'] + [col for col in df_merged.columns if col != 'ID']
df_merged = df_merged[cols]
df_merged = df_merged.loc[:, ~(df_merged.isna() | (df_merged == 0)).all()]

# Split the data into multiple files, with a maximum number of columns per file.
num_files = (df_merged.shape[1] + max_columns - 1) // max_columns

for i in range(num_files):
    start_col = i * max_columns
    end_col = min((i + 1) * max_columns, df_merged.shape[1])
    subset = df_merged.iloc[:, start_col:end_col]
    subset.to_excel(f'./Parallel/PMB/{modality}_merged_data_part_{i + 1}.xlsx', index=False)

print(f"Data has been split into {num_files} files.")


