import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Original_datasets/train.csv", index_col="Id")

def get_nulls(df):
    null_columns = df.isnull().sum()
    null_columns = null_columns[null_columns>0]
    df_nulls = pd.DataFrame({"null_count": null_columns, "null_percent": (null_columns/df.shape[0])*100}).sort_values(by='null_count', ascending=False)
    print(df_nulls)
    plt.figure(figsize=(6, 13))
    sns.heatmap(df.isnull().T, yticklabels=df.columns, xticklabels=False, cbar=False, cmap='viridis')
    return df_nulls

df_nulls = get_nulls(df)

# 'PoolQC' - 

# 'MiscFeature' - 
# 'Alley' - 
# 'Fence' - 
# 'FireplaceQu' - 
# 'LotFrontage' - 
# 'GarageType' - 
# 'GarageYrBlt' - 
# 'GarageFinish' - 
# 'GarageQual' - 
# 'GarageCond' - 
# 'BsmtExposure' - 
# 'BsmtFinType2' - 
# 'BsmtFinType1' - 
# 'BsmtCond' - 
# 'BsmtQual' - 
# 'MasVnrArea' - 
# 'MasVnrType' - 
# 'Electrical' - 
