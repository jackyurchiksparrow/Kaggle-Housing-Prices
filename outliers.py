import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Post_EDA_datasets/train.csv")
df_test = pd.read_csv("Post_EDA_datasets/test.csv")

numerical_rows = df.select_dtypes(include='number')
numerical_rows.columns
continuous_rows = numerical_rows[['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'SalePrice', 'HouseAge', 'RemodelledYearsAgo', 'BsmtFinSF', 'ShedSF']]

for column in continuous_rows.columns.values:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Create a subplot with two axes
    
     # Plot using boxplot
    axes[0].boxplot(continuous_rows[column].dropna())
    axes[0].set_xlabel(column)
    axes[0].set_title('Boxplot')
   
     # Plot using sns.displot
    sns.histplot(data=continuous_rows, x=column, ax=axes[1], kde=True, bins=31)
    axes[1].set_xlabel(column)
    axes[1].set_title('Histogram with KDE')
    
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

def get_IQR_outliears(column):
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = column[(column < lower_bound) | (column > upper_bound)]
    return outliers_iqr

def get_z_scores_outliers(column, threshold = 3):
    z_scores = (column - np.mean(column)) / np.std(column)
    outliers_zscore = column[np.abs(z_scores) > threshold]
    return outliers_zscore

LotFrontage = df['LotFrontage'].dropna()
# normal distribution so we'll use z-scores
outliers_zscore = get_z_scores_outliers(LotFrontage)
df.drop(outliers_zscore.index, inplace=True)

LotArea = df['LotArea']
plt.hist(LotArea, bins=51)
plt.xlabel('LotArea')
plt.show()
# normal distribution so we'll use z-scores
outliers_zscore = get_z_scores_outliers(LotArea)
df.drop(outliers_zscore.index, inplace=True)

MasVnrArea = df['MasVnrArea']
# exponential distribution so we'll use IQR
IQR_outliers = get_IQR_outliears(MasVnrArea)
# no outliers

BsmtUnfSF = df['BsmtUnfSF']
# truncated normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(BsmtUnfSF)
df.drop(outliers_zscore.index, inplace=True)


TotalBsmtSF = df['TotalBsmtSF']
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(TotalBsmtSF)
df.drop(outliers_zscore.index, inplace=True)

firstFlrSF = df['1stFlrSF']
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(firstFlrSF)
df.drop(outliers_zscore.index, inplace=True)

# we will exclude zeroes as we need it to determine if a house
# has second floor or not; they cannot be considered outliers
secondFlrSF = df[df['2ndFlrSF']>0]['2ndFlrSF']

plt.hist(secondFlrSF, bins=31)
plt.xlabel('2ndFlrSF')
plt.show()
# with zeroes exluded the distribution is normal
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(secondFlrSF)
df.drop(outliers_zscore.index, inplace=True)

# we will exclude zeroes as we need it to determine if a house
# has finished low quality feet or not; they cannot be considered outliers
LowQualFinSF = df[df['LowQualFinSF']>0]['LowQualFinSF']
# after excluding zeroes it has only 25 rows, thus, is highly imbalanced
# exclude it (the pattern on the plot is not obvious as well as its distribution)
continuous_rows.drop('LowQualFinSF', inplace=True, axis=1)
df.drop('LowQualFinSF', inplace=True, axis=1)
df_test.drop('LowQualFinSF', inplace=True, axis=1)



















