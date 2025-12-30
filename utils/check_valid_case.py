import os
import pandas as pd

# 读取CSV文件
csv_file_path = r"F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\filter_clinical.csv"
df = pd.read_csv(csv_file_path)

# 获取ID列
id_column = df['ID']

# 获取包含子文件夹的目录路径
directory_path = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'

# 获取所有子文件夹的名称
subfolders = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

# 创建一个列表来存储匹配的行
matching_rows = []

# 遍历ID列，检查是否有匹配的子文件夹
for id_value in id_column:
    # 将ID转换为字符串以便进行字符串操作
    id_str = str(id_value)

    # 检查是否有子文件夹名以ID开头
    for subfolder in subfolders:
        if subfolder.startswith(id_str):
            matching_rows.append(df[df['ID'] == id_value])
            break

# 将匹配的行合并为一个DataFrame
matching_df = pd.concat(matching_rows)

# 将结果保存到新的CSV文件
output_csv_path = r"F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\valid_case.csv"
matching_df.to_csv(output_csv_path, index=False)

print(f"Matching rows have been saved to {output_csv_path}")