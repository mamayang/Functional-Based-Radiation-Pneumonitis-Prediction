import pandas as pd
import numpy as np


def count_valid_columns(file_path, exclude_columns):
    # 读取xlsx文件
    df = pd.read_csv(file_path)

    # 要排除的列
    columns_to_exclude = exclude_columns

    # 找出全为0或nan的列
    zero_nan_columns = []
    for col in df.columns:
        if col not in columns_to_exclude:
            if df[col].isna().all() or (df[col] == 0).all():
                zero_nan_columns.append(col)

    # 计算有效列数
    total_columns = len(df.columns)
    excluded_columns = len(columns_to_exclude)
    zero_nan_column_count = len(zero_nan_columns)
    valid_columns = total_columns - excluded_columns - zero_nan_column_count

    print(f"总列数: {total_columns}")
    print(f"指定排除的列数: {excluded_columns}")
    print(f"全为0或nan的列数: {zero_nan_column_count}")
    print(f"有效列数: {valid_columns}")
    print("\n全为0或nan的列名:")
    for col in zero_nan_columns:
        print(col)

    return valid_columns

csv_path = r"E:\Code_test\pneumonia\Code\dataset\for_radiomics\Dose_result\function0.3\dose_masked_low_perfusion_dosimetric.csv"
# 使用示例
exclude_cols = ['Age', 'Dose', 'Gender_1', 'Gender_2', 'Smoke_1',
                'Smoke_2', 'Treatment_0', 'Treatment_1', 'Treatment_2', 'Grade']
result = count_valid_columns(csv_path, exclude_cols)