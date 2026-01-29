import pandas as pd
import numpy as np
from scipy import stats

# Read CSV file
df = pd.read_csv('../dataset/for_radiomics/valid_case.csv')

# Create pneumonia grouping (0: no pneumonia, 1: pneumonia)
df['pneumonia_group'] = (df['clinical_feature_pneumonia'] >= 2).astype(int)

# Exclude ID-related columns
columns_to_analyze = df.columns.drop(['ID', 'clinical_feature_pneumonia', 'pneumonia_group'])

results = {}

for col in columns_to_analyze:
    # Determine whether the variable is continuous by the number of unique values
    unique_values = df[col].nunique()

    if unique_values >= 5:  # Continuous variable
        group0 = df[df['pneumonia_group'] == 0][col]
        group1 = df[df['pneumonia_group'] == 1][col]

        # Normality test
        _, p_shapiro0 = stats.shapiro(group0)
        _, p_shapiro1 = stats.shapiro(group1)

        # If both groups follow normal distribution (p > 0.05), use t-test; otherwise use Mann-Whitney U
        if p_shapiro0 > 0.05 and p_shapiro1 > 0.05:
            stat, p_value = stats.ttest_ind(group0, group1)
            test_type = "t-test"
        else:
            stat, p_value = stats.mannwhitneyu(group0, group1, alternative='two-sided')
            test_type = "Mann-Whitney U"

        results[col] = {
            'p_value': p_value,
            'test_type': test_type,
            'total': f"mean={round(np.mean(df[col]), 2)}, std={round(np.std(df[col]), 2)}",
            'mean_no_pneumonia': np.mean(group0),
            'mean_pneumonia': np.mean(group1),
            'std_no_pneumonia': np.std(group0),
            'std_pneumonia': np.std(group1)
        }

    else:  # Categorical variable
        # Count each category in both groups
        counts_no_pneumonia = df[df['pneumonia_group'] == 0][col].value_counts()
        counts_pneumonia = df[df['pneumonia_group'] == 1][col].value_counts()

        # Create contingency table
        contingency_table = pd.crosstab(df[col], df['pneumonia_group'])

        # Use Fisher's exact test for 2x2 tables; otherwise use Chi-square test
        if contingency_table.shape == (2, 2):
            _, p_value = stats.fisher_exact(contingency_table, alternative='two-sided')
            test_type = "Fisher exact"
        else:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            test_type = "Chi-square"

        results[col] = {
            'p_value': p_value,
            'test_type': test_type,
            'total': "N/A",
            'counts_no_pneumonia': dict(counts_no_pneumonia),
            'counts_pneumonia': dict(counts_pneumonia)
        }

# Convert results into a more readable format
formatted_results = {}
for col, result in results.items():
    if 'mean_no_pneumonia' in result:  # Continuous variable
        formatted_results[col] = {
            'p_value': round(result['p_value'], 4),
            'test_type': result['test_type'],
            'total': result['total'],
            'no_pneumonia': f"mean={round(result['mean_no_pneumonia'], 2)}, std={round(result['std_no_pneumonia'], 2)}",
            'pneumonia': f"mean={round(result['mean_pneumonia'], 2)}, std={round(result['std_pneumonia'], 2)}"
        }
    else:  # Categorical variable
        formatted_results[col] = {
            'p_value': round(result['p_value'], 4),
            'test_type': result['test_type'],
            'total': result['total'],
            'no_pneumonia': result['counts_no_pneumonia'],
            'pneumonia': result['counts_pneumonia']
        }

# Convert to DataFrame and save
results_df = pd.DataFrame.from_dict(formatted_results, orient='index')
results_df.to_csv('statistical_analysis_results.csv')

# Print results
print("Statistical Analysis Results:")
print(results_df)

# Read CSV file
df = pd.read_csv('../dataset/for_radiomics/valid_case.csv')

# Create pneumonia grouping (0: no pneumonia, 1: pneumonia)
df['pneumonia_group'] = (df['clinical_feature_pneumonia'] >= 2).astype(int)

# Create contingency table
contingency_table = pd.crosstab(df['TNM_IIIA_ 0_IIIB_1'], df['pneumonia_group'])
print("Contingency table:")
print(contingency_table)

# 1. Fisher's exact test
_, fisher_p = stats.fisher_exact(contingency_table)
print(f"\n1. Fisher's exact test p-value: {fisher_p}")

# 2. Chi-square test (with Yates correction)
chi2_yates, p_yates, _, _ = stats.chi2_contingency(contingency_table, correction=True)
print(f"2. Chi-square test with Yates correction p-value: {p_yates}")

# 3. Chi-square test (without correction)
chi2, p_chi2, _, _ = stats.chi2_contingency(contingency_table, correction=False)
print(f"3. Chi-square test without correction p-value: {p_chi2}")

# Calculate percentage in each group
print("\nPercentages in each group:")
percent_table = pd.crosstab(df['Cigrate'], df['pneumonia_group'], normalize='columns') * 100
print(percent_table)

# Calculate sample size in each group
print("\nSample size in each group:")
print(df['pneumonia_group'].value_counts())