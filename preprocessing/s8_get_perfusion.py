import SimpleITK as sitk
import os


def resample_image(image, new_spacing=(1.0, 1.0, 1.0)):
    # 获取原始图像的尺寸和spacing
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 计算新的尺寸，保证图像内容的完整性
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]

    # 设置重采样器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    # 重采样图像
    return resampler.Execute(image)


def process_folder(src_folder, dst_folder):
    for subdir, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith("function.nii"):
                src_file_path = os.path.join(subdir, file)
                image = sitk.ReadImage(src_file_path)

                # 重采样图像
                resampled_image = resample_image(image)

                # 构建目标文件路径
                rel_path = os.path.relpath(subdir, src_folder)
                dst_subdir = os.path.join(dst_folder, rel_path)
                os.makedirs(dst_subdir, exist_ok=True)
                dst_file_path = os.path.join(dst_subdir, "perfusion.mha")

                # 保存重采样后的图像
                sitk.WriteImage(resampled_image, dst_file_path)
                print(f"Resampled and copied '{src_file_path}' to '{dst_file_path}'")


# 使用示例
#原数据中有两个数据集包含有perfusion.mha，所以需要跑两个文件夹
# source_directory = r'E:\1-Original_dataset\pneumonia\with_perfusion\2021_All_Image_Esophagus_Data'
source_directory = r'E:\1-Original_dataset\pneumonia\with_perfusion\Residual_Lung'  # PATH A
target_directory = r'E:\Code_test\pneumonia\Code\dataset\for_radiomics\Function_for_radiomics'  # PATH B

process_folder(source_directory, target_directory)

