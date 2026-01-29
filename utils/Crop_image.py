import os
import SimpleITK as sitk


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


def process_folder(folder_path):
    # 获取mask文件路径
    mask_path = os.path.join(folder_path, 'resampled_lung_mask.mha')
    if not os.path.exists(mask_path):
        print(f"Mask file not found in {folder_path}")
        return

    # 首先处理非mask文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.mha') and filename != 'resampled_lung_mask.mha':
            input_path = os.path.join(folder_path, filename)
            try:
                crop_by_mask(input_path, mask_path)
                print(f"Successfully cropped and replaced {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # 最后处理mask文件本身
    try:
        crop_by_mask(mask_path, mask_path)
        print("Successfully cropped and replaced resampled_lung_mask.mha")
    except Exception as e:
        print(f"Error processing mask: {str(e)}")


def process_all_folders(root_path):
    # 遍历所有文件夹
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path):
            print(f"\nProcessing folder: {folder}")
            process_folder(folder_path)


# 使用示例
root_directory = r"F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin"  # 替换为您的根目录路径
process_all_folders(root_directory)