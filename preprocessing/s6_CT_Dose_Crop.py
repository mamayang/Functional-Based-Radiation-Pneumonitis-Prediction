



# Dose is already aligned, so no need to realign, only need to change spacing

import SimpleITK as sitk

import os

def find_matching_files(dose_folder_path, ct_folder_path):
    matching_files = []

    # Get all subfolders in the dose folder
    dose_subdirs = [d for d in os.listdir(dose_folder_path) if os.path.isdir(os.path.join(dose_folder_path, d))]

    for subdir in dose_subdirs:
        dose_subdir_path = os.path.join(dose_folder_path, subdir)
        ct_subdir_path = os.path.join(ct_folder_path, subdir)

        # Check whether a subfolder with the same name exists in the CT folder
        if os.path.isdir(ct_subdir_path):
            dose_file = os.path.join(dose_subdir_path, 'Dose.mha')
            ct_file = os.path.join(ct_subdir_path, 'ct.mha')

            # Check whether both files exist
            if os.path.isfile(dose_file) and os.path.isfile(ct_file):
                matching_files.append((dose_file, ct_file))
            else:
                if not os.path.isfile(dose_file):
                    print(f"Warning: Dose.mha not found in {dose_subdir_path}")
                if not os.path.isfile(ct_file):
                    print(f"Warning: ct.mha not found in {ct_subdir_path}")

    return matching_files

def crop_by_mask(input_path, mask_path):
    # Read input image and mask
    image = sitk.ReadImage(input_path)
    mask = sitk.ReadImage(mask_path)

    # Get the non-zero bounding box of the mask
    filter = sitk.LabelStatisticsImageFilter()
    filter.Execute(mask, mask)
    bbox = filter.GetBoundingBox(1)  # 假设mask中的标签值为1

    # Extract coordinates of the bounding box
    lower = bbox[0::2]
    upper = bbox[1::2]

    # Crop image
    size = [u - l for l, u in zip(lower, upper)]
    cropped = sitk.RegionOfInterest(image, size, lower)

    # Overwrite the original file directly
    sitk.WriteImage(cropped, input_path)

def process_folder(matching_files):
    """
    Process the specified folders and adjust lung_mask.mha to match the size and spacing of Dose.mha.
    """
    for dose_path, ct_path in matching_files:


        # lung_mask_path = os.path.join(subdir, "lung_mask.mha")
        # Get parent directory
        dose_parent_parent = os.path.dirname(dose_path)
        lung_mask_path = os.path.join(dose_parent_parent, 'lung_mask.mha' )
        if os.path.exists(dose_path) and os.path.exists(lung_mask_path):
            # Read images
            dose_img = sitk.ReadImage(dose_path)
            lung_mask_img = sitk.ReadImage(lung_mask_path)
            ct_img = sitk.ReadImage(ct_path)
            # Resample lung_mask.mha
            resampled_dose_img = resample_to_reference(dose_img, ct_img)
            resampled_dose_img_path = os.path.join(dose_parent_parent, "dose_registrated_with_ct.mha")
            sitk.WriteImage(resampled_dose_img, resampled_dose_img_path)
            resampled_lung_mask = resample_to_reference(lung_mask_img,resampled_dose_img)
            # Save resampled image
            resampled_lung_mask_path = os.path.join(dose_parent_parent, "resampled_lung_mask.mha")
            sitk.WriteImage(resampled_lung_mask, resampled_lung_mask_path)
            print(f"Resampled lung_mask.mha saved to {resampled_lung_mask_path}")
        else:
            print(f"Dose.mha or lung_mask.mha not found in {dose_parent_parent}")


# Specify paths
parent_fold = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics'
dose_folder_path = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\Dose_map_radiomics\Dose_map'
ct_folder_path = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'
matching_files = find_matching_files(dose_folder_path, ct_folder_path)
# Process folders
process_folder(matching_files)