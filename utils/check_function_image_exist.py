import os


def find_missing_files(main_folder):
    required_files = ['high_perfusion_mask.mha', 'high_ventilation_mask.mha',
                      'low_perfusion_mask.mha', 'low_ventilation_mask.mha']
    missing_files_dict = {}

    # Traverse all subfolders in the main folder
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        # Ensure this is a folder
        if os.path.isdir(subfolder_path):
            missing_files = []

            # Check whether each required file exists
            for file in required_files:
                file_path = os.path.join(subfolder_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)

            # If there are missing files, add them to the dictionary
            if missing_files:
                missing_files_dict[subfolder] = missing_files

    return missing_files_dict


# Use the function
main_folder_path = r'E:\Code_test\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'  # Please replace with your main folder path
missing_files_result = find_missing_files(main_folder_path)

# Print results
if missing_files_result:
    print("The following subfolders are missing some required files:")
    for subfolder, missing_files in missing_files_result.items():
        print(f"{subfolder}:")
        for file in missing_files:
            print(f"  - {file}")
        print()  # Add blank line for better readability
else:
    print("All subfolders contain all required files.")

print(f"In total, {len(missing_files_result)} subfolders are missing one or more required files.")