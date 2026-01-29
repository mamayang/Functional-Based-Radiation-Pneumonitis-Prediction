import pandas as pd
import ast
import os
import re
def filter_and_save_data(ids_path, data_file, output_file):
    try:
        # Load the IDs from the first Excel file
        ids_list = os.listdir(ids_path)
        ids_int_list = [int(re.match(r'\d+', s).group()) for s in ids_list]
        # Load the data from the second Excel file
        detailed_data = pd.read_csv(data_file,sep=',')

        # Filter rows where the ID is in the ids_list
        filtered_data = detailed_data[detailed_data['ID'].isin(ids_int_list)]

        # Selecting specific columns
        columns_needed = {
            'ID': 'ID',
            'age': 'Age',
            'Man_1_Woman_2': 'Gender',
            'Cigrate': 'Smoke',
            'CCRT1_ChemoRadio_2_RadioChemo_0': 'Treatment',
            'Dose': 'Dose',
            'clinical_feature_pneumonia': 'Grade',
        }
        final_data = filtered_data[list(columns_needed.keys())]
        modify_file = final_data.rename(columns=columns_needed, inplace=False)
        modify_file['Grade'] = (modify_file['Grade'] >= 2).astype(int)
        modify_file['Dose'] = pd.to_numeric(modify_file['Dose'], errors='coerce')
        # Delete rows containing null values
        modify_file = modify_file.dropna()
        # Save the filtered data to a CSV file
        modify_file.to_csv(output_file, index=False)
        print("Data has been successfully saved to", output_file)
    except Exception as e:
        print(f"An error occurred: {e}")

# Adjust the file paths as necessary
ids_path = r'E:\Code_test\pneumonia\Code\dataset\for_radiomics\Dose_map_radiomics\Dose_map_preprocessed'
data_file =r'E:\Code_test\pneumonia\clinical_feature_pneumonia.csv'
output_file = r'E:\Code_test\pneumonia\Code\dataset\select_clinical.csv'


filter_and_save_data(ids_path, data_file, output_file)
