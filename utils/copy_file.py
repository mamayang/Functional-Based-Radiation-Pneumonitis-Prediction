import SimpleITK as sitk
import os

def process_folder(src_folder, dst_folder):
    for subdir, dirs, files in os.walk(src_folder):
        for file in files:
            if file == "dose_registrated_with_ct.mha":
                src_file_path = os.path.join(subdir, file)
                image = sitk.ReadImage(src_file_path)

                # 重采样图像
                # 构建目标文件路径
                rel_path = os.path.relpath(subdir, src_folder)
                dst_subdir = os.path.join(dst_folder, rel_path)
                os.makedirs(dst_subdir, exist_ok=True)
                dst_file_path = os.path.join(dst_subdir, "dose_registrated_with_ct.mha")

                # 保存重采样后的图像
                sitk.WriteImage(image, dst_file_path)
                print(f"Processed and copied '{src_file_path}' to '{dst_file_path}'")




# 设定源目录和目标目录
source_directory = r'E:\Code_test\pneumonia\Code\dataset\for_radiomics\Dose_map_radiomics\Dose_map'  # PATH A
target_directory = r'E:\Code_test\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'

# 执行函数
process_folder(source_directory , target_directory )