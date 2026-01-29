import pandas as pd
import numpy as np
import os


def normalize_column(column):
    # Avoid division by zero
    if column.max() == column.min():
        return column
    else:
        # Normalize to the range [-1, 1]
        return 2 * (column - column.min()) / (column.max() - column.min()) - 1


def normalize_csv(file_path):
    # Read Excel file
    df = pd.read_excel(file_path)

    # Columns that do not need normalization
    exclude_cols = ['ID', 'Age', 'Dose', 'Gender_1', 'Gender_2', 'Smoke_1', 'Smoke_2',
                    'Treatment_0', 'Treatment_1', 'Treatment_2', 'Grade']

    # Get columns that need normalization
    normalize_cols = [col for col in df.columns if col not in exclude_cols]

    # Create a new DataFrame by copying original data
    normalized_df = df.copy()

    # Normalize each column
    for col in normalize_cols:
        normalized_df[col] = normalize_column(df[col])

    # Generate output file name
    base_name = os.path.splitext(file_path)[0]
    output_path = f"{base_name}_normalized.csv"

    # Save normalized data
    normalized_df.to_csv(output_path, index=False)
    print(f"Processed: {file_path} -> {output_path}")


def process_all_csvs(directory="."):
    # Process all Excel files in the specified directory
    for file in os.listdir(directory):
        if file.endswith(".xlsx"):
            file_path = os.path.join(directory, file)
            normalize_csv(file_path)


# Run main program
if __name__ == "__main__":
    process_all_csvs(directory=r'F:\Mayang_Code\RP\Radition pneumonitis\Code\pneumonia\Code\Parallel\PMB')