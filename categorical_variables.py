import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("Post_nulls_datasets/train.csv", na_filter=False)
df_test = pd.read_csv("Post_nulls_datasets/test.csv", na_filter=False)
df_nulls_for_iterative_imputer = pd.read_csv("Post_nulls_datasets/train_iterative_imputer_nulls.csv", na_filter=False)
df_test_nulls_for_iterative_imputer = pd.read_csv("Post_nulls_datasets/test_iterative_imputer_nulls.csv", na_filter=False)

# 1. Determine which columns require categorical processing
# all non-numbers for sure
categorical_cols = df.select_dtypes(include='object')
numeric_cols = df.select_dtypes(include='number')

sns.histplot(data=df, x='YearBuilt')
# research all years further

categorical_cols = pd.concat([categorical_cols, df[['MSSubClass', 'OverallQual', 'OverallCond']]], axis=1)






























