import os
import pandas as pd


def list_subfolders(path):
    """ 列出指定路径下的所有子文件夹 """
    return [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]


def compare_folders(folder1, folder2):
    """ 比较两个文件夹的子文件夹并找出folder2中缺失的子文件夹 """
    subfolders1 = set(list_subfolders(folder1))
    subfolders2 = set(list_subfolders(folder2))

    # 找出folder2中缺失的子文件夹
    missing_folders = subfolders1 - subfolders2
    return list(missing_folders)


def save_to_csv(missing_folders, output_file):
    """ 保存缺失的子文件夹到CSV文件 """
    df = pd.DataFrame(missing_folders, columns=['Missing Perfusion cases'])
    df.to_csv(output_file, index=False)
    print(f"缺失的子文件夹已保存到 {output_file}")


# 设置文件夹路径
folder1 = r'E:\Code_test\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'
folder2 = r'E:\Code_test\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_processed'

# 设置输出CSV文件的路径
output_csv = 'missing_folders.csv'

# 执行比较和保存
missing_folders = compare_folders(folder1, folder2)
save_to_csv(missing_folders, output_csv)