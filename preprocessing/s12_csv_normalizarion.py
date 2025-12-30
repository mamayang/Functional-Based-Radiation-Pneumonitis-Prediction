import pandas as pd
import numpy as np
import os


def normalize_column(column):
    # 避免除零错误
    if column.max() == column.min():
        return column
    else:
        # 归一化到[-1, 1]范围
        return 2 * (column - column.min()) / (column.max() - column.min()) - 1


def normalize_csv(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 不需要归一化的列
    exclude_cols = ['ID', 'Age', 'Dose', 'Gender_1', 'Gender_2', 'Smoke_1', 'Smoke_2',
                    'Treatment_0', 'Treatment_1', 'Treatment_2', 'Grade']

    # 获取需要归一化的列
    normalize_cols = [col for col in df.columns if col not in exclude_cols]

    # 创建新的DataFrame，先复制原始数据
    normalized_df = df.copy()

    # 对每一列进行归一化
    for col in normalize_cols:
        normalized_df[col] = normalize_column(df[col])

    # 生成输出文件名
    base_name = os.path.splitext(file_path)[0]
    output_path = f"{base_name}_normalized.csv"

    # 保存归一化后的数据
    normalized_df.to_csv(output_path, index=False)
    print(f"Processed: {file_path} -> {output_path}")


def process_all_csvs(directory="."):
    # 处理指定目录下的所有Excel文件
    for file in os.listdir(directory):
        if file.endswith(".xlsx"):
            file_path = os.path.join(directory, file)
            normalize_csv(file_path)


# 执行主程序
if __name__ == "__main__":
    process_all_csvs(directory=r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\Parallel\PMB')