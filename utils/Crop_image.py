import os
import SimpleITK as sitk


def crop_by_mask(input_path, mask_path):
    # Read input image and mask
    image = sitk.ReadImage(input_path)
    mask = sitk.ReadImage(mask_path)

    # Get the non-zero bounding box of the mask
    filter = sitk.LabelStatisticsImageFilter()
    filter.Execute(mask, mask)
    bbox = filter.GetBoundingBox(1)  # Assume the label value in the mask is 1

    # Extract bounding box coordinates
    lower = bbox[0::2]
    upper = bbox[1::2]

    # Crop image
    size = [u - l for l, u in zip(lower, upper)]
    cropped = sitk.RegionOfInterest(image, size, lower)

    # Overwrite the original file directly
    sitk.WriteImage(cropped, input_path)


def process_folder(folder_path):
    # Get mask file path
    mask_path = os.path.join(folder_path, 'resampled_lung_mask.mha')
    if not os.path.exists(mask_path):
        print(f"Mask file not found in {folder_path}")
        return

    # First process non-mask files
    for filename in os.listdir(folder_path):
        if filename.endswith('.mha') and filename != 'resampled_lung_mask.mha':
            input_path = os.path.join(folder_path, filename)
            try:
                crop_by_mask(input_path, mask_path)
                print(f"Successfully cropped and replaced {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Finally process the mask file itself
    try:
        crop_by_mask(mask_path, mask_path)
        print("Successfully cropped and replaced resampled_lung_mask.mha")
    except Exception as e:
        print(f"Error processing mask: {str(e)}")


def process_all_folders(root_path):
    # Traverse all folders
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path):
            print(f"\nProcessing folder: {folder}")
            process_folder(folder_path)


# Usage example
root_directory = r"F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin"  # Replace with your root directory path
process_all_folders(root_directory)
