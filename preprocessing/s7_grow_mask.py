# Please use this after radiomics preprocessing of GTV is completed
import os
import SimpleITK as sitk
def dilate_mask(input_path, output_path, expand_mm=3):
    # Read mask
    mask = sitk.ReadImage(input_path, sitk.sitkUInt8)

    # Get image resolution (spacing)
    spacing = mask.GetSpacing()

    # Calculate kernel size required for dilation
    kernel_size = [int(expand_mm / s + 0.5) for s in spacing]

    # Create spherical dilation kernel
    kernel = sitk.sitkBall
    radius = kernel_size
    dilated_mask = sitk.BinaryDilate(mask, radius, kernel)

    # Save processed mask
    sitk.WriteImage(dilated_mask, output_path)


def process_folder(root_dir):
    # Traverse all subdirectories under the root directory
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


# Specify the root directory path containing subfolders
root_dir = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'
process_folder(root_dir)
