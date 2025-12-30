#请在radiomics已经预处理完GTV后使用
import os
import SimpleITK as sitk
def dilate_mask(input_path, output_path, expand_mm=3):
    # 读取mask
    mask = sitk.ReadImage(input_path, sitk.sitkUInt8)

    # 获取图像的分辨率
    spacing = mask.GetSpacing()

    # 计算扩展所需的核大小
    kernel_size = [int(expand_mm / s + 0.5) for s in spacing]

    # 创建球形扩展核
    kernel = sitk.sitkBall
    radius = kernel_size
    dilated_mask = sitk.BinaryDilate(mask, radius, kernel)

    # 保存处理后的mask
    sitk.WriteImage(dilated_mask, output_path)


def process_folder(root_dir):
    # 遍历根目录下的所有子目录
    for subdir, dirs, files in os.walk(root_dir):
        for dirname in dirs:
            subfolder_path = os.path.join(subdir, dirname)
            input_path = os.path.join(subfolder_path, 'resampled_gtv_mask.mha')
            output_path = os.path.join(subfolder_path, 'grow_mask.mha')

            if os.path.exists(input_path):
                print(f"Processing {input_path}...")
                dilate_mask(input_path, output_path)
                print(f"Output saved to {output_path}")
            else:
                print(f"No mask found in {subfolder_path}")


# 指定含有子文件夹的根目录路径
root_dir = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'
process_folder(root_dir)
