import os


def find_missing_files(main_folder):
    required_files = ['high_perfusion_mask.mha', 'high_ventilation_mask.mha',
                      'low_perfusion_mask.mha', 'low_ventilation_mask.mha']
    missing_files_dict = {}

    # 遍历主文件夹中的所有子文件夹
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        # 确保这是一个文件夹
        if os.path.isdir(subfolder_path):
            missing_files = []

            # 检查每个所需文件是否存在
            for file in required_files:
                file_path = os.path.join(subfolder_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)

            # 如果有缺失的文件，将它们添加到字典中
            if missing_files:
                missing_files_dict[subfolder] = missing_files

    return missing_files_dict


# 使用函数
main_folder_path = r'E:\Code_test\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'  # 请替换为您的主文件夹路径
missing_files_result = find_missing_files(main_folder_path)

# 打印结果
if missing_files_result:
    print("以下子文件夹缺少一些所需文件：")
    for subfolder, missing_files in missing_files_result.items():
        print(f"{subfolder}:")
        for file in missing_files:
            print(f"  - {file}")
        print()  # 为了更好的可读性添加空行
else:
    print("所有子文件夹都包含所有所需文件。")

print(f"总共有 {len(missing_files_result)} 个子文件夹缺少一个或多个所需文件。")