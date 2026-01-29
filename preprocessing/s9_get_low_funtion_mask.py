import SimpleITK as sitk
import os
import numpy as np
from options import parse_option

opt = parse_option(print_option=True)

def process_mha_files(input_base_directory, output_base_directory):
    # Traverse all subdirectories under the input base directory
    for subdir, dirs, files in os.walk(input_base_directory):
        for file in files:
            if file == "Ven_1500_7.mha":
                # Construct the full path of the input file
                input_file_path = os.path.join(subdir, file)

                # Read .mha file
                image = sitk.ReadImage(input_file_path)
                array = sitk.GetArrayFromImage(image)

                # Calculate thresholds for the top 30% and 0.1% of the value range
                lower_threshold = array.max() * 0.001
                upper_threshold = array.max() * opt.function_threshold

                # Create 0/1 mask
                mask = (array > lower_threshold) & (array <= upper_threshold)
                mask_image = sitk.GetImageFromArray(mask.astype(int))
                mask_image.CopyInformation(image)  # Preserve original image metadata

                # Construct the full path of the output file (keeping the same subdirectory structure)
                relative_path = os.path.relpath(subdir, input_base_directory)

                last_part = os.path.basename(relative_path)
                output_subdir = os.path.join(output_base_directory, last_part)
                os.makedirs(output_subdir, exist_ok=True)  # Create output directory (if it does not exist)
                output_path = os.path.join(output_subdir, "low_ventilation_mask.mha")

                # Save mask as new .mha file
                sitk.WriteImage(mask_image, output_path)
                print(f"Mask saved to {output_path}")

            if file == "moved_back_perfusion.nii":
                # Construct the full path of the input file
                input_file_path = os.path.join(subdir, file)

                # Read .mha file
                image = sitk.ReadImage(input_file_path)
                array = sitk.GetArrayFromImage(image)

                # Calculate thresholds for the top 30% and 0.1% of the value range
                lower_threshold = array.max() * 0.001
                upper_threshold = array.max() * opt.function_threshold

                # Create 0/1 mask
                mask = (array > lower_threshold) & (array <= upper_threshold)
                mask_image = sitk.GetImageFromArray(mask.astype(int))
                mask_image.CopyInformation(image)  # Preserve original image metadata

                # Construct the full path of the output file (keeping the same subdirectory structure)
                relative_path = os.path.relpath(subdir, input_base_directory)

                last_part = os.path.basename(relative_path)
                output_subdir = os.path.join(output_base_directory, last_part)
                os.makedirs(output_subdir, exist_ok=True)  # Create output directory (if it does not exist)
                output_path = os.path.join(output_subdir, "low_perfusion_mask.mha")

                # Save mask as new .mha file
                sitk.WriteImage(mask_image, output_path)
                print(f"Mask saved to {output_path}")



# Specify input base directory and output base directory that contain subfolders
input_base_directory = '../dataset/for_radiomics/CT_radiomics/functional image'
# output_base_directory = os.path.join('../dataset/for_radiomics/CT_radiomics/CT_origin',str(opt.function_threshold))
output_base_directory = '../dataset/for_radiomics/CT_radiomics/CT_origin'
os.makedirs(output_base_directory, exist_ok=True)
process_mha_files(input_base_directory, output_base_directory)