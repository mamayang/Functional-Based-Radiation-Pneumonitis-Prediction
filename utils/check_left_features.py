import pandas as pd
import numpy as np


def count_valid_columns(file_path, exclude_columns):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Columns to exclude
    columns_to_exclude = exclude_columns

    # Find columns that are all 0 or NaN
    zero_nan_columns = []
    for col in df.columns:
        if col not in columns_to_exclude:
            if df[col].isna().all() or (df[col] == 0).all():
                zero_nan_columns.append(col)

    # Calculate number of valid columns
    total_columns = len(df.columns)
    excluded_columns = len(columns_to_exclude)
    zero_nan_column_count = len(zero_nan_columns)
    valid_columns = total_columns - excluded_columns - zero_nan_column_count

    print(f"Total number of columns: {total_columns}")
    print(f"Number of explicitly excluded columns: {excluded_columns}")
    print(f"Number of columns that are all 0 or NaN: {zero_nan_column_count}")
    print(f"Number of valid columns: {valid_columns}")
    print("\nNames of columns that are all 0 or NaN:")
    for col in zero_nan_columns:
        print(col)

    return valid_columns

csv_path = r"E:\Code_test\pneumonia\Code\dataset\for_radiomics\Dose_result\function0.3\dose_masked_low_perfusion_dosimetric.csv"
# 使用示例
exclude_cols = ['Age', 'Dose', 'Gender_1', 'Gender_2', 'Smoke_1',
                'Smoke_2', 'Treatment_0', 'Treatment_1', 'Treatment_2', 'Grade']
result = count_valid_columns(csv_path, exclude_cols)