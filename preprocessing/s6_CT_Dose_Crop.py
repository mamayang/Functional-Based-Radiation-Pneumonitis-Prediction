



#Dose 本身就是对齐的不需要对齐，只需要改space

import SimpleITK as sitk

import os

def find_matching_files(dose_folder_path, ct_folder_path):
    matching_files = []

    # 获取 dose 文件夹中的所有子文件夹
    dose_subdirs = [d for d in os.listdir(dose_folder_path) if os.path.isdir(os.path.join(dose_folder_path, d))]

    for subdir in dose_subdirs:
        dose_subdir_path = os.path.join(dose_folder_path, subdir)
        ct_subdir_path = os.path.join(ct_folder_path, subdir)

        # 检查 CT 文件夹中是否存在相同名称的子文件夹
        if os.path.isdir(ct_subdir_path):
            dose_file = os.path.join(dose_subdir_path, 'Dose.mha')
            ct_file = os.path.join(ct_subdir_path, 'ct.mha')

            # 检查两个文件是否都存在
            if os.path.isfile(dose_file) and os.path.isfile(ct_file):
                matching_files.append((dose_file, ct_file))
            else:
                if not os.path.isfile(dose_file):
                    print(f"Warning: Dose.mha not found in {dose_subdir_path}")
                if not os.path.isfile(ct_file):
                    print(f"Warning: ct.mha not found in {ct_subdir_path}")

    return matching_files

def crop_by_mask(input_path, mask_path):
    # 读取输入图像和mask
    image = sitk.ReadImage(input_path)
    mask = sitk.ReadImage(mask_path)

    # 获取mask的非零边界框
    filter = sitk.LabelStatisticsImageFilter()
    filter.Execute(mask, mask)
    bbox = filter.GetBoundingBox(1)  # 假设mask中的标签值为1

    # 提取边界框坐标
    lower = bbox[0::2]
    upper = bbox[1::2]

    # 裁剪图像
    size = [u - l for l, u in zip(lower, upper)]
    cropped = sitk.RegionOfInterest(image, size, lower)

    # 直接覆盖原文件
    sitk.WriteImage(cropped, input_path)

def process_folder(matching_files):
    """
    处理指定文件夹，调整其中的 lung_mask.mha 以匹配 Dose.mha 的大小和像素间距。
    """
    for dose_path, ct_path in matching_files:


        # lung_mask_path = os.path.join(subdir, "lung_mask.mha")
        # 获取上一级目录
        dose_parent_parent = os.path.dirname(dose_path)
        lung_mask_path = os.path.join(dose_parent_parent, 'lung_mask.mha' )
        if os.path.exists(dose_path) and os.path.exists(lung_mask_path):
            # 读取图像
            dose_img = sitk.ReadImage(dose_path)
            lung_mask_img = sitk.ReadImage(lung_mask_path)
            ct_img = sitk.ReadImage(ct_path)
            # 重采样 lung_mask.mha
            resampled_dose_img = resample_to_reference(dose_img, ct_img)
            resampled_dose_img_path = os.path.join(dose_parent_parent, "dose_registrated_with_ct.mha")
            sitk.WriteImage(resampled_dose_img, resampled_dose_img_path)
            resampled_lung_mask = resample_to_reference(lung_mask_img,resampled_dose_img)
            # 保存重采样后的图像
            resampled_lung_mask_path = os.path.join(dose_parent_parent, "resampled_lung_mask.mha")
            sitk.WriteImage(resampled_lung_mask, resampled_lung_mask_path)
            print(f"Resampled lung_mask.mha saved to {resampled_lung_mask_path}")
        else:
            print(f"Dose.mha or lung_mask.mha not found in {dose_parent_parent}")


# 指定路径
parent_fold = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics'
dose_folder_path = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\Dose_map_radiomics\Dose_map'
ct_folder_path = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'
matching_files = find_matching_files(dose_folder_path, ct_folder_path)
# 处理文件夹
process_folder(matching_files)