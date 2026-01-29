import os
import pandas as pd

# Read CSV file
csv_file_path = r"F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\filter_clinical.csv"
df = pd.read_csv(csv_file_path)

# Get ID column
id_column = df['ID']

# Get the directory path that contains subfolders
directory_path = r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\for_radiomics\CT_radiomics\CT_origin'

# Get names of all subfolders
subfolders = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

# Create a list to store matching rows
matching_rows = []

# Traverse the ID column and check for matching subfolders
for id_value in id_column:
    # Convert ID to string for string operations
    id_str = str(id_value)

    # Check whether any subfolder name starts with the ID
    for subfolder in subfolders:
        if subfolder.startswith(id_str):
            matching_rows.append(df[df['ID'] == id_value])
            break

# Concatenate matching rows into a single DataFrame
matching_df = pd.concat(matching_rows)

# Save the result to a new CSV file
output_csv_path = r"F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\dataset\valid_case.csv"
matching_df.to_csv(output_csv_path, index=False)

print(f"Matching rows have been saved to {output_csv_path}")