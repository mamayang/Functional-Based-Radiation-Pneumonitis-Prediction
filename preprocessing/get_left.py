import os
import pandas as pd

# 读取包含ID的CSV文件
csv_path = "../dataset/for_radiomics/clinical_feature_pneumonia.csv"
df = pd.read_csv(csv_path)

# 文件夹路径
folder_path = "../dataset/for_radiomics/Dose_map_radiomics/Dose_map"

# 获取文件夹中的所有子文件夹名称，并去掉可能的'a'后缀
subfolder_names = [name.rstrip('a') for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

# 找到匹配的行
matching_rows = df[df['ID'].astype(str).isin(subfolder_names)]

# 将匹配的行保存到新的CSV文件
output_path = "../dataset/for_radiomics/sift_pneumonia.csv"
matching_rows.to_csv(output_path, index=False)

print(f"找到 {len(matching_rows)} 个匹配项")
print(f"结果已保存到 {output_path}")