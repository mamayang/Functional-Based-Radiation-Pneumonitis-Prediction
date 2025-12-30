import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Step 2: Load the CSV file into a DataFrame
df = pd.read_csv(r'E:\Code_test\pneumonia\Code\dataset\select_clinical.csv')
y = df['Grade']
ids = df['ID']
X = df.drop(['ID', 'Grade'], axis=1)
categorical_features = ['Gender', 'Smoke', 'Treatment']
numerical_features = ['Age','Dose']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X_preprocessed = preprocessor.fit_transform(X)
cat_ohe_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_columns = numerical_features + list(cat_ohe_columns)
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=all_columns)
X_preprocessed_df['Grade'] = y.values
X_preprocessed_df.insert(0, 'ID', ids)
# Step 5: Optionally save the filtered DataFrame to a new CSV file
X_preprocessed_df.to_csv(r'E:\Code_test\pneumonia\Code\dataset\filter_clinical.csv', index=False)