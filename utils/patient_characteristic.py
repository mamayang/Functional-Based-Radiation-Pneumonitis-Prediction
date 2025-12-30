import pandas as pd
import numpy as np
from scipy import stats

# Read CSV file
df = pd.read_csv('../dataset/for_radiomics/valid_case.csv')

# 创建肺炎分组（0：无肺炎，1：有肺炎）
df['pneumonia_group'] = (df['clinical_feature_pneumonia'] >= 2).astype(int)

# 排除ID列
columns_to_analyze = df.columns.drop(['ID', 'clinical_feature_pneumonia', 'pneumonia_group'])

results = {}

for col in columns_to_analyze:
    # 通过不同值的数量来判断是否为连续变量
    unique_values = df[col].nunique()

    if unique_values >= 5:  # 连续变量
        group0 = df[df['pneumonia_group'] == 0][col]
        group1 = df[df['pneumonia_group'] == 1][col]

        # 正态性检验
        _, p_shapiro0 = stats.shapiro(group0)
        _, p_shapiro1 = stats.shapiro(group1)

        # 如果两组都是正态分布(p > 0.05)，使用t检验；否则使用Mann-Whitney U
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

    else:  # 分类变量
        # 计算各组中每个类别的数量
        counts_no_pneumonia = df[df['pneumonia_group'] == 0][col].value_counts()
        counts_pneumonia = df[df['pneumonia_group'] == 1][col].value_counts()

        # 创建列联表
        contingency_table = pd.crosstab(df[col], df['pneumonia_group'])

        # 对于2x2的表格使用Fisher精确检验，其他情况使用卡方检验
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

# 将结果转换为更易读的格式
formatted_results = {}
for col, result in results.items():
    if 'mean_no_pneumonia' in result:  # 连续变量
        formatted_results[col] = {
            'p_value': round(result['p_value'], 4),
            'test_type': result['test_type'],
            'total': result['total'],
            'no_pneumonia': f"mean={round(result['mean_no_pneumonia'], 2)}, std={round(result['std_no_pneumonia'], 2)}",
            'pneumonia': f"mean={round(result['mean_pneumonia'], 2)}, std={round(result['std_pneumonia'], 2)}"
        }
    else:  # 分类变量
        formatted_results[col] = {
            'p_value': round(result['p_value'], 4),
            'test_type': result['test_type'],
            'total': result['total'],
            'no_pneumonia': result['counts_no_pneumonia'],
            'pneumonia': result['counts_pneumonia']
        }

# 转换为DataFrame并保存
results_df = pd.DataFrame.from_dict(formatted_results, orient='index')
results_df.to_csv('statistical_analysis_results.csv')

# 打印结果
print("Statistical Analysis Results:")
print(results_df)

# Read CSV file
df = pd.read_csv('../dataset/for_radiomics/valid_case.csv')

# 创建肺炎分组（0：无肺炎，1：有肺炎）
df['pneumonia_group'] = (df['clinical_feature_pneumonia'] >= 2).astype(int)

# 创建列联表
contingency_table = pd.crosstab(df['TNM_IIIA_ 0_IIIB_1'], df['pneumonia_group'])
print("Contingency table:")
print(contingency_table)

# 1. Fisher精确检验
_, fisher_p = stats.fisher_exact(contingency_table)
print(f"\n1. Fisher's exact test p-value: {fisher_p}")

# 2. 卡方检验（带Yates校正）
chi2_yates, p_yates, _, _ = stats.chi2_contingency(contingency_table, correction=True)
print(f"2. Chi-square test with Yates correction p-value: {p_yates}")

# 3. 卡方检验（不带校正）
chi2, p_chi2, _, _ = stats.chi2_contingency(contingency_table, correction=False)
print(f"3. Chi-square test without correction p-value: {p_chi2}")

# 计算每组的百分比
print("\nPercentages in each group:")
percent_table = pd.crosstab(df['Cigrate'], df['pneumonia_group'], normalize='columns') * 100
print(percent_table)

# 计算样本量
print("\nSample size in each group:")
print(df['pneumonia_group'].value_counts())