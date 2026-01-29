import os
import shutil

def delete_specific_files(root_folder):
    files_to_delete = [
        "dose_registrated_with_ct.mha",
        'resampled_lung_mask.mha',
        'dose_masked_high_perfusion_mask.mha',
        'dose_masked_high_ventilation_mask.mha'
    ]

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename in files_to_delete:
                file_path = os.path.join(foldername, filename)
                print(f"Deleting: {file_path}")
                os.remove(file_path)

# 使用示例
root_folder = r"F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\Dose_map_radiomics\Dose_map"  # 替换为您的根文件夹路径
delete_specific_files(root_folder)