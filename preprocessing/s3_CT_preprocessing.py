import SimpleITK as sitk
import os


def resample_to_reference(moving_image, reference_image):
    """
    Resample the moving image so that it matches the size and spacing of the reference image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.AffineTransform(moving_image.GetDimension()))
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampled_image = resampler.Execute(moving_image)
    return resampled_image


def process_folder(folder_path):
    """
    Process the specified folder and adjust lung_mask.mha to match the size and spacing of ct.mha.
    """
    for subdir, dirs, files in os.walk(folder_path):
        for dirname in dirs:
            ct_path = os.path.join(subdir,dirname, "ct.mha")
            lung_mask_path = os.path.join(subdir,dirname, "lung_mask.mha")
            gtv_mask_path = os.path.join(subdir,dirname, "gtv.mha")
            if os.path.exists(ct_path) and os.path.exists(lung_mask_path):
                # Read images
                ct_img = sitk.ReadImage(ct_path)
                lung_mask_img = sitk.ReadImage(lung_mask_path)

                # Resample lung_mask.mha
                resampled_lung_mask = resample_to_reference(lung_mask_img, ct_img)

                # Save resampled image
                resampled_lung_mask_path = os.path.join(subdir,dirname, "resampled_lung_mask.mha")
                sitk.WriteImage(resampled_lung_mask, resampled_lung_mask_path)
                print(f"Resampled lung_mask.mha saved to {resampled_lung_mask_path}")
            else:
                print(f"ct.mha or lung_mask.mha not found in {subdir}")

            if os.path.exists(ct_path) and os.path.exists(gtv_mask_path):
                # Read images
                ct_img = sitk.ReadImage(ct_path)
                gtv_mask_img = sitk.ReadImage(gtv_mask_path)

                # Resample gtv_mask.mha
                resampled_gtv_mask = resample_to_reference(gtv_mask_img, ct_img)

                # Save resampled image
                resampled_gtv_mask_path = os.path.join(subdir,dirname, "resampled_gtv_mask.mha")
                sitk.WriteImage(resampled_gtv_mask, resampled_gtv_mask_path)
                print(f"Resampled gtv_mask.mha saved to {resampled_gtv_mask_path}")
            else:
                print(f"ct.mha or gtv_mask.mha not found in {resampled_gtv_mask_path}")

# Specify path
folder_path = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'

# Process folder
process_folder(folder_path)