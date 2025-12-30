import SimpleITK as sitk
import os
import numpy as np
from options import parse_option

opt = parse_option(print_option=True)

def process_mha_files(input_base_directory, output_base_directory):
    # 遍历输入基目录下的所有子目录
    for subdir, dirs, files in os.walk(input_base_directory):
        for file in files:
            if file == "Ven_1500_7.mha":
                # 构建输入文件的完整路径
                input_file_path = os.path.join(subdir, file)

                # 读取.mha文件
                image = sitk.ReadImage(input_file_path)
                array = sitk.GetArrayFromImage(image)

                # 计算值域的前30%和0.1%的阈值
                lower_threshold = array.max() * 0.001
                upper_threshold = array.max() * opt.function_threshold

                # 创建0,1 mask
                mask = (array > lower_threshold) & (array <= upper_threshold)
                mask_image = sitk.GetImageFromArray(mask.astype(int))
                mask_image.CopyInformation(image)  # 保留原图像的元数据信息

                # 构建输出文件的完整路径（保持相同的子目录结构）
                relative_path = os.path.relpath(subdir, input_base_directory)

                last_part = os.path.basename(relative_path)
                output_subdir = os.path.join(output_base_directory, last_part)
                os.makedirs(output_subdir, exist_ok=True)  # 创建输出目录（如果不存在）
                output_path = os.path.join(output_subdir, "low_ventilation_mask.mha")

                # 保存mask为新的.mha文件
                sitk.WriteImage(mask_image, output_path)
                print(f"Mask saved to {output_path}")

            if file == "moved_back_perfusion.nii":
                # 构建输入文件的完整路径
                input_file_path = os.path.join(subdir, file)

                # 读取.mha文件
                image = sitk.ReadImage(input_file_path)
                array = sitk.GetArrayFromImage(image)

                # 计算值域的前30%和0.1%的阈值
                lower_threshold = array.max() * 0.001
                upper_threshold = array.max() * opt.function_threshold

                # 创建0,1 mask
                mask = (array > lower_threshold) & (array <= upper_threshold)
                mask_image = sitk.GetImageFromArray(mask.astype(int))
                mask_image.CopyInformation(image)  # 保留原图像的元数据信息

                # 构建输出文件的完整路径（保持相同的子目录结构）
                relative_path = os.path.relpath(subdir, input_base_directory)

                last_part = os.path.basename(relative_path)
                output_subdir = os.path.join(output_base_directory, last_part)
                os.makedirs(output_subdir, exist_ok=True)  # 创建输出目录（如果不存在）
                output_path = os.path.join(output_subdir, "low_perfusion_mask.mha")

                # 保存mask为新的.mha文件
                sitk.WriteImage(mask_image, output_path)
                print(f"Mask saved to {output_path}")



# 指定包含子文件夹的输入基目录和输出基目录
input_base_directory = '../dataset/for_radiomics/CT_radiomics/functional image'
# output_base_directory = os.path.join('../dataset/for_radiomics/CT_radiomics/CT_origin',str(opt.function_threshold))
output_base_directory = '../dataset/for_radiomics/CT_radiomics/CT_origin'
os.makedirs(output_base_directory, exist_ok=True)
process_mha_files(input_base_directory, output_base_directory)