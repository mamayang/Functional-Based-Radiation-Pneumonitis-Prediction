import SimpleITK as sitk
import os


def resample_image(image, new_spacing=(1.0, 1.0, 1.0)):
    # Get original image size and spacing
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Calculate new size to ensure completeness of image content
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]

    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    # Resample image
    return resampler.Execute(image)


def process_folder(src_folder, dst_folder):
    for subdir, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith("RTDoseAll.mha"):
                src_file_path = os.path.join(subdir, file)
                image = sitk.ReadImage(src_file_path)

                # Resample image
                resampled_image = resample_image(image)

                # Build target file path
                rel_path = os.path.relpath(subdir, src_folder)
                dst_subdir = os.path.join(dst_folder, rel_path)
                os.makedirs(dst_subdir, exist_ok=True)
                dst_file_path = os.path.join(dst_subdir, "dose.mha")

                # Save resampled image
                sitk.WriteImage(resampled_image, dst_file_path)
                print(f"Resampled and copied '{src_file_path}' to '{dst_file_path}'")


# Usage example

source_directory = r'G:\1_Original_image\radiation pneumonitis\All_Image_Esophagus_Data'  # PATH A
target_directory = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'  # PATH B

process_folder(source_directory, target_directory)
