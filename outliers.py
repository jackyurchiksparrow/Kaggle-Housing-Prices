import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Post_EDA_datasets/train.csv")
df_test = pd.read_csv("Post_EDA_datasets/test.csv")

numerical_rows = df.select_dtypes(include='number')
numerical_rows.columns
continuous_rows = numerical_rows[['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd','Fireplaces', 'GarageYrBlt', 'GarageCars','GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'SalePrice', 'HouseAge', 'RemodelledYearsAgo', 'BsmtFinSF', 'ShedSF']]

      
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
sns.histplot(LotArea, bins=71)
plt.xlabel('LotArea')
plt.show()
# normal distribution so we'll use z-scores
outliers_zscore = get_z_scores_outliers(LotArea)
df.drop(outliers_zscore.index, inplace=True)


YearBuilt = df['YearBuilt']
# multiple normal distribution (mixture distribution) - z-scores
outliers_zscore = get_z_scores_outliers(YearBuilt)
df.drop(outliers_zscore.index, inplace=True)


YearRemodAdd = df['YearRemodAdd']
# multiple normal distribution (mixture distribution) - z-scores
outliers_zscore = get_z_scores_outliers(YearRemodAdd)
# no outliers

# further below, we will exclude zeroes when they can't be considered
# outliers; in this case we need them to determine if a house
# has a masonry or not
MasVnrArea = df[df['MasVnrArea']>0]['MasVnrArea']
# 582/1427 non zero values; we have enough data to create a binary column 
# 1 - a house has a masonry, 0 - not. We will further make a research for 
# multicolinearity
df['isMasonry'] = df['MasVnrArea'].apply(lambda x: 1 if x>0 else 0)
df_test['isMasonry'] = df_test['MasVnrArea'].apply(lambda x: 1 if x>0 else 0)

sns.histplot(MasVnrArea, bins=31)
plt.xlabel('MasVnrArea')
plt.show()
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(MasVnrArea)
df.drop(outliers_zscore.index, inplace=True)


BsmtUnfSF = df[df['BsmtUnfSF']>0]['BsmtUnfSF']
sns.histplot(BsmtUnfSF, bins=21)
plt.xlabel('BsmtUnfSF')
plt.show()
# truncated normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(BsmtUnfSF)
df.drop(outliers_zscore.index, inplace=True)


TotalBsmtSF = df[df['TotalBsmtSF']>0]['TotalBsmtSF']
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(TotalBsmtSF)
df.drop(outliers_zscore.index, inplace=True)


firstFlrSF = df[df['1stFlrSF']>0]['1stFlrSF']
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(firstFlrSF)
df.drop(outliers_zscore.index, inplace=True)


secondFlrSF = df[df['2ndFlrSF']>0]['2ndFlrSF']
# 599/1394 non zero values; we will create a binary column 1 - a house has
# a second floor, 0 - not. We will further make a research for multicolinearity
df['isSecondFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)
df_test['isSecondFloor'] = df_test['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)
sns.histplot(secondFlrSF, bins=31)
plt.xlabel('2ndFlrSF')
plt.show()
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(secondFlrSF)
df.drop(outliers_zscore.index, inplace=True)


LowQualFinSF = df[df['LowQualFinSF']>0]['LowQualFinSF']
# after excluding zeroes it has only 24 rows, thus, is highly imbalanced
# exclude it (the pattern on the plot is not obvious as well as its distribution)
df.drop('LowQualFinSF', inplace=True, axis=1)
df_test.drop('LowQualFinSF', inplace=True, axis=1)


GrLivArea = df['GrLivArea']
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(GrLivArea)
df.drop(outliers_zscore.index, inplace=True)


BsmtFullBath = df['BsmtFullBath']
# 819 houses without a full bathroom in the basement, 
# 554 - with 1, 10 - with 2, 1 - with 3
# we will add another binary column, 
# where 0 - not a full bathroom in the bsmt,  1 - one and more bathrooms
df['isFullBathBsmt'] = df['BsmtFullBath'].apply(lambda x: 1 if x>=1 else 0)
df_test['isFullBathBsmt'] = df_test['BsmtFullBath'].apply(lambda x: 1 if x>=1 else 0)
# drop used columns
df.drop('BsmtFullBath', inplace=True, axis=1)
df_test.drop('BsmtFullBath', inplace=True, axis=1)


BsmtHalfBath = df['BsmtHalfBath']
# 1306 houses without a half bathroom in the basement,
# 76 - with 1, 2 - with 2
# the data is highly imbalanced, better to drop
df.drop('BsmtHalfBath', inplace=True, axis=1)
df_test.drop('BsmtHalfBath', inplace=True, axis=1)


FullBath = df['FullBath']
# 718 houses with 2 bathrooms, 641 - with 1, 17 - with 3, 8 - with 0
# we will add another binary column, 
# where 0 - not a full bathroom,  1 - one and more bathrooms
df['isFullBath'] = df['FullBath'].apply(lambda x: 1 if x>=1 else 0)
df_test['isFullBath'] = df_test['FullBath'].apply(lambda x: 1 if x>=1 else 0)
# drop used columns
df.drop('FullBath', inplace=True, axis=1)
df_test.drop('FullBath', inplace=True, axis=1)


HalfBath = df['HalfBath']
# 873 houses without half bathrooms, 499 - with 1, 12 - with 2
# we will add another binary column, 
# where 0 - not a half bathroom,  1 - one and more half bathrooms
df['isHalfBath'] = df['HalfBath'].apply(lambda x: 1 if x>=1 else 0)
df_test['isHalfBath'] = df_test['HalfBath'].apply(lambda x: 1 if x>=1 else 0)
# drop used columns
continuous_rows.drop('HalfBath', inplace=True, axis=1)
df.drop('HalfBath', inplace=True, axis=1)
df_test.drop('HalfBath', inplace=True, axis=1)


TotRmsAbvGrd = df['TotRmsAbvGrd']
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(TotRmsAbvGrd)
df.drop(outliers_zscore.index, inplace=True)


Fireplaces = df['Fireplaces']
# 673 houses with no fireplaces, 601 - with 1, 88 - with 2, 4 - with 3
# we will add another binary column, 
# where 0 - not a fireplace,  1 - one and more fireplaces
df['isFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x>=1 else 0)
df_test['isFireplace'] = df_test['Fireplaces'].apply(lambda x: 1 if x>=1 else 0)
# drop used columns
df.drop('Fireplaces', inplace=True, axis=1)
df_test.drop('Fireplaces', inplace=True, axis=1)


GarageYrBlt = df['GarageYrBlt']
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(GarageYrBlt)
df.drop(outliers_zscore.index, inplace=True)


GarageCars = df['GarageCars']
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(GarageCars)
df.drop(outliers_zscore.index, inplace=True)


GarageArea = df[df['GarageArea']>0]['GarageArea']
# 1286/1362 non zero values; imbalanced, thus, no need to create a binary column
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(GarageArea)
df.drop(outliers_zscore.index, inplace=True)


WoodDeckSF = df[df['WoodDeckSF']>0]['WoodDeckSF']
# 635/1354 non zero values; we will create a binary column 1 - a house has
# a wood deck, 0 - not. We will further make a research for multicolinearity
df['isWoodDeck'] = df['WoodDeckSF'].apply(lambda x: 1 if x>0 else 0)
df_test['isWoodDeck'] = df_test['WoodDeckSF'].apply(lambda x: 1 if x>0 else 0)
sns.histplot(WoodDeckSF, bins=31)
plt.xlabel('WoodDeckSF')
plt.show()
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(WoodDeckSF)
df.drop(outliers_zscore.index, inplace=True)


OpenPorchSF = df[df['OpenPorchSF']>0]['OpenPorchSF']
# 720/1343 non zero values; we will create a binary column 1 - a house has
# an open porch, 0 - not. We will further make a research for multicolinearity
df['isOpenPorch'] = df['OpenPorchSF'].apply(lambda x: 1 if x>0 else 0)
df_test['isOpenPorch'] = df_test['OpenPorchSF'].apply(lambda x: 1 if x>0 else 0)
sns.histplot(OpenPorchSF, bins=31)
plt.xlabel('OpenPorchSF')
plt.show()
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(OpenPorchSF)
df.drop(outliers_zscore.index, inplace=True)


EnclosedPorch = df[df['EnclosedPorch']>0]['EnclosedPorch']
# 190/1332 non zero values; imbalanced, thus, no need to create a binary column
sns.histplot(EnclosedPorch, bins=21)
plt.xlabel('EnclosedPorch')
plt.show()
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(EnclosedPorch)
# no outliers


ThreeSsnPorch = df[df['3SsnPorch']>0]['3SsnPorch']
# after excluding zeroes it has only 24 rows, thus, is highly imbalanced
# exclude it
df.drop('3SsnPorch', inplace=True, axis=1)
df_test.drop('3SsnPorch', inplace=True, axis=1)


ScreenPorch = df[df['ScreenPorch']>0]['ScreenPorch']
# 103/1332 non zero values; imbalanced, thus, no need to create a binary column
sns.histplot(ScreenPorch, bins=21)
plt.xlabel('ScreenPorch')
plt.show()
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(ScreenPorch)
# doubtful impact, but it has only 2 outliers so we will allow it to be
df.drop(outliers_zscore.index, inplace=True)


PoolArea = df[df['PoolArea']>0]['PoolArea']
# after excluding zeroes it has only 2 rows, thus, is highly imbalanced
# and not useful in predicting; exclude it
df.drop('PoolArea', inplace=True, axis=1)
df_test.drop('PoolArea', inplace=True, axis=1)


SalePrice = df['SalePrice']
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(SalePrice)
df.drop(outliers_zscore.index, inplace=True)


HouseAge = df['HouseAge']
# multiple normal distribution (mixture distribution) - z-scores
outliers_zscore = get_z_scores_outliers(HouseAge)
df.drop(outliers_zscore.index, inplace=True)


RemodelledYearsAgo = df[df['RemodelledYearsAgo']>0]['RemodelledYearsAgo']
# 609/1311 non zero values; we will create a binary column 1 - a house was
# remodelled, 0 - was'nt. We will further make a research for multicolinearity
sns.histplot(RemodelledYearsAgo, bins=31, kde=True)
plt.xlabel('RemodelledYearsAgo')
plt.show()
# exponential distribution - IQR method
outliers_IQR = get_IQR_outliears(RemodelledYearsAgo)
df.drop(outliers_IQR.index, inplace=True)


BsmtFinSF = df[df['BsmtFinSF']>0]['BsmtFinSF']
sns.histplot(BsmtFinSF, bins=31, kde=True)
plt.xlabel('BsmtFinSF')
plt.show()
# normal distribution - z-scores
outliers_zscore = get_z_scores_outliers(BsmtFinSF)
df.drop(outliers_zscore.index, inplace=True)


ShedSF = df[df['ShedSF']>0]['ShedSF']
# after excluding zeroes it has only 41 rows, thus, is highly imbalanced
# but might have a predicting power as a binary variable (isShed) so we'll leave it
# but exclude the continuous one
df.drop('ShedSF', inplace=True, axis=1)
df_test.drop('ShedSF', inplace=True, axis=1)

df.to_csv('Post_outliers_datasets/train.csv', index=False)
df_test.to_csv('Post_outliers_datasets/test.csv', index=False)







