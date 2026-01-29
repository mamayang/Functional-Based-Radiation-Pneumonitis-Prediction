import os
import pandas as pd

# Read the CSV file containing IDs
csv_path = "../dataset/for_radiomics/clinical_feature_pneumonia.csv"
df = pd.read_csv(csv_path)

# Folder path
folder_path = "../dataset/for_radiomics/Dose_map_radiomics/Dose_map"

# Get all subfolder names in the folder and remove possible 'a' suffix
subfolder_names = [name.rstrip('a') for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

# Find matching rows
matching_rows = df[df['ID'].astype(str).isin(subfolder_names)]

# Save the matching rows to a new CSV file
output_path = "../dataset/for_radiomics/sift_pneumonia.csv"
matching_rows.to_csv(output_path, index=False)

print(f"Found {len(matching_rows)} matches")
print(f"Results have been saved to {output_path}")