import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, zscore

# 读取 CSV 文件
df = pd.read_csv(r"E:\Code_test\pneumonia\Code\dataset\for_radiomics\valid_case.csv")
df['Dose'] = pd.to_numeric(df['Dose'], errors='coerce')
# 删除包含空值的行
df.dropna(inplace=True)

# 创建新的分类列
df['clinical_feature_pneumonia_class'] = (df['clinical_feature_pneumonia'] >= 2).astype(int)
df['age_class'] = (df['age'] >= 55).astype(int)
# 进行 Chi-squared 测试的列
chi_squared_columns = ['Man_1_Woman_2', 'Cigrate', 'TNM_IIIA_ 0_IIIB_1', 'CCRT1_ChemoRadio_2_RadioChemo_0']

# 进行 Mann-Whitney U 检验的列
mann_whitney_columns = ['age','Dose']
df[mann_whitney_columns] = df[mann_whitney_columns].apply(zscore)
# 进行 Chi-squared 测试
for col in chi_squared_columns:
    contingency_table = pd.crosstab(df[col], df['clinical_feature_pneumonia_class'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-squared test for {col}:")
    print(f"Chi2: {chi2}, P-value: {p}, Degrees of Freedom: {dof}")
    print("Expected Frequencies:")
    print(expected)
    print('-' * 50)

# 进行 Mann-Whitney U 检验
for col in mann_whitney_columns:
    group1 = df[df['clinical_feature_pneumonia_class'] == 0][col]
    group2 = df[df['clinical_feature_pneumonia_class'] == 1][col]
    u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"Mann-Whitney U test for {col}:")
    print(f"U-statistic: {u_stat}, P-value: {p_val}")
    print('-' * 50)