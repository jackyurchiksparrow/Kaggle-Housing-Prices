import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = pd.read_csv("Post_outliers_datasets/train.csv")
df_test = pd.read_csv("Post_outliers_datasets/test.csv")

def get_nulls(df):
    null_columns = df.isnull().sum()
    null_columns = null_columns[null_columns>0]
    df_nulls = pd.DataFrame({"null_count": null_columns, "null_percent": (null_columns/df.shape[0])*100}).sort_values(by='null_count', ascending=False)
    print(df_nulls)
    plt.figure(figsize=(6, 13))
    sns.heatmap(df.isnull().T, yticklabels=df.columns, xticklabels=False, cbar=False, cmap='viridis')
    return df_nulls


df_nulls = get_nulls(df)
df_test_nulls = get_nulls(df_test)

# PLAN
# 1. Handle obvious nulls using statistics or logic.
# 2. Determine the data that needs a more advanced approach to be completed.
# a. Separate the variables: Split dataset into two parts - one with the variables 
# that need imputation and another with the variables that are complete.
# b. Deal with categorical values to be able to apply IterativeImputer.
# c. Impute missing values: Apply IterativeImputer on the part of the dataset containing 
# the variables with null values. This will use the observed relationships in the 
# variables to impute the missing values in the target variables.
# d. Merge the datasets: Once the missing values are imputed, merge the imputed dataset 
# with the complete dataset to have a complete dataset with all the variables.
# 5. Proceed with further analysis or modeling: Now that you have a complete dataset, 
# you can proceed with your analysis or build a predictive model using the imputed values.


# ------------------------- Handling obvious nulls -------------------------


# 'PoolQC' - more than 99% missing data, only 2 non-null values, exclude it
df.drop('PoolQC', axis=1, inplace=True)
df_test.drop('PoolQC', axis=1, inplace=True)


# 'Alley' - 93.9% missing values, 1229/1308
# according to the docs 'NA' should be no alley access, so we might replace nulls with NA
# but it is extremely imbalanced and unlikely to help in predicting, exclude it
df.drop('Alley', axis=1, inplace=True)
df_test.drop('Alley', axis=1, inplace=True)


# 'Fence' - 80.19% missing values, 1049 / 1308	
# according to the docs 'NA' should be no fence, but we don't have a single 'NA' and 80% of the dat
# is missing. It is better to exclude it
df.drop('Fence', axis=1, inplace=True)
df_test.drop('Fence', axis=1, inplace=True)


# MasVnrType - 0.45% missing values, 6/1308
# 'None' should be 'no masonry', apply this
df.loc[(df['isMasonry']==0) & (df['MasVnrType'].isnull()==True), 'MasVnrType'] = 'None' # 6


# same for df_test
# MasVnrType - 1.09% missing values, 16/1459
df_test.loc[(df_test['isMasonry']==0) & (df_test['MasVnrType'].isnull()==True), 'MasVnrType'] = 'None' # 15

MasVnrType_nulls = df_test[df_test['MasVnrType'].isnull()==True]

MasVnrType_nulls.loc[MasVnrType_nulls.index, 'FireplaceQu'] = 'NA' # because isFireplace is 0 there
df_test.loc[MasVnrType_nulls.index, 'FireplaceQu'] = 'NA' # because isFireplace is 0 there

df_test_nulls_for_iterative_imputer = df_test.loc[MasVnrType_nulls.index]
df_test.drop(MasVnrType_nulls.index, inplace=True)





df_nulls = get_nulls(df)
df_test_nulls = get_nulls(df_test)





# 'FireplaceQu' - 50.38% missing values, 659 / 1308	
# according to the docs 'NA' should be no fireplace
# we will assign 'NA' to absent fireplaces
df[(df['isFireplace']==0) & (df['FireplaceQu'].isnull() == True)] # 659
df.loc[(df['isFireplace']==0) & (df['FireplaceQu'].isnull() == True), 'FireplaceQu'] = 'NA' # 657

# same for df_test
# 'FireplaceQu' - 50.0% missing values, 729 / 1308	
df_test[(df_test['isFireplace']==0) & (df_test['FireplaceQu'].isnull() == True)] # 729
df_test.loc[(df_test['isFireplace']==0) & (df_test['FireplaceQu'].isnull() == True), 'FireplaceQu'] = 'NA' # 729


# 'LotFrontage' - 17.73% missing values, 232 / 1308	
# a continuous variable with a normal distribution, so we can replace nulls with mean
imputer_mean = SimpleImputer(strategy='mean')
df['LotFrontage'] = imputer_mean.fit_transform(df[['LotFrontage']])
df_test['LotFrontage'] = imputer_mean.fit_transform(df_test[['LotFrontage']])



# 'GarageType' - 5.65% missing values, 74 / 1308
# 'GarageYrBlt' - 5.65% missing values, 74 / 1308
# 'GarageFinish' - 5.65% missing values, 74 / 1308	
# 'GarageQual' - 5.65% missing values, 74 / 1308	
# 'GarageCond' - 5.65% missing values, 74 / 1308
# --- 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond' all
# have the exact same amount of null values (5.65% missing values, 74 / 1308) +
# documentation states about 'NA' being  no garage and it's the only value missing.
# Thus, we can imply all of them are values for "no garage"; let's check that
df[(df['GarageArea']==0) & (df['GarageType'].isnull()==True) & (df['GarageYrBlt'].isnull()==True) & (df['GarageFinish'].isnull()==True) & (df['GarageQual'].isnull()==True) & (df['GarageCond'].isnull()==True)] # 74
# proved
df['GarageYrBlt'].fillna(0, inplace=True)
columns_to_fill = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in columns_to_fill:
    df[col].fillna("NA", inplace=True)

# same for df_test
# GarageFinish - 5.34% missing valus, 78/1458	
# GarageYrBlt - 5.34% missing valus, 78/1458	
# GarageCond - 5.34% missing valus, 78/1458
# GarageQual - 5.34% missing valus, 78/1458
# GarageType - 5.21% missing valus, 76/1458
garages_nulls_common = df_test[(df_test['GarageArea']==0) & (df_test['GarageType'].isnull()==True) & (df_test['GarageYrBlt'].isnull()==True) & (df_test['GarageFinish'].isnull()==True) & (df_test['GarageQual'].isnull()==True) & (df_test['GarageCond'].isnull()==True)] # 76
diff_ind = df_test[df_test['GarageFinish'].isnull()==True].index.symmetric_difference(garages_nulls_common.index)
diff = df_test.loc[diff_ind]

# according to the specified garage type these houses have garage(s), so we will save them for imputing
df_test_nulls_for_iterative_imputer = pd.concat([diff, df_test_nulls_for_iterative_imputer])
df_test.drop(diff_ind, inplace=True)

df_test['GarageYrBlt'].fillna(0, inplace=True)
for col in columns_to_fill:
    df_test[col].fillna("NA", inplace=True)


# make a binary column for easier classification
df['isGarage'] = df['GarageType'].apply(lambda x: 0 if x=='NA' else 1)
df_test['isGarage'] = df_test['GarageType'].apply(lambda x: 0 if x=='NA' else 1)
df_test_nulls_for_iterative_imputer['isGarage'] = df_test_nulls_for_iterative_imputer['GarageType'].apply(lambda x: 0 if x=='NA' else 1)


# 'BsmtExposure' - 	2.82% missing values, 37 / 1308
# according to the docs 'NA' should be no basement, we will check if there any others
# create a column to characterize a basement by its presence or absence
df['TotalBsmtSF'].isnull().any() # False
df['isBasement'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df.loc[(df['isBasement']==0) & (df['BsmtExposure'].isnull()==True), 'BsmtExposure'] = 'NA' # 36
# unclassified ones add to the df_nulls_for_iterative_imputer
df_nulls_for_iterative_imputer = df[df['BsmtExposure'].isnull()==True]
df = df.drop(df_nulls_for_iterative_imputer.index)


# same for df_test
# BsmtExposure - 3.02% missing values, 44 / 1456	
# the TotalBsmtSF has 1 null in df_test, all basement columns are nan there
# so we cannot say if it has basement or not, not obvious
df_test['isBasement'] = np.nan # assigning nans to all isBasement rows
# appending the row with nan basement parameters to impute further
df_test_nulls_for_iterative_imputer = pd.concat([df_test[df_test['TotalBsmtSF'].isnull()==True], df_test_nulls_for_iterative_imputer])
# remove it from the test dataset
df_test = df_test.drop(df_test[df_test['TotalBsmtSF'].isnull()==True].index)


# setting other values of isBasement after removing TotalBsmtSF nan
df_test['isBasement'] = df_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)      
df_test.loc[(df_test['isBasement']==0) & (df_test['BsmtExposure'].isnull()==True), 'BsmtExposure'] = 'NA' # 41
# others are not obvious
unobv_bsmt_exposure_nulls = df_test[df_test['BsmtExposure'].isnull()==True] # 2
df_test_nulls_for_iterative_imputer = pd.concat([unobv_bsmt_exposure_nulls, df_test_nulls_for_iterative_imputer])
df_test = df_test.drop(unobv_bsmt_exposure_nulls.index)





df_nulls = get_nulls(df)
df_test_nulls = get_nulls(df_test)







# 'BsmtFinType1' - 2.75% missing values, 36 / 1307	
# 'BsmtFinType2' - 2.75% missing values, 36 / 1307
# 'BsmtCond' - 2.75% missing values, 36 / 1307	
# 'BsmtQual' - 2.75% missing values, 36 / 1307
# --- now, the 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual' all has the exact
# same amount of null values. So we can imply all of them are values for "no basement"
# let's check that
df[(df['isBasement']==0) & (df['BsmtFinType1'].isnull()==True) & (df['BsmtFinType2'].isnull()==True) & (df['BsmtCond'].isnull()==True) & (df['BsmtQual'].isnull()==True)] # 36
# proved
columns_to_fill = ['BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual']
for col in columns_to_fill:
    df[col].fillna("NA", inplace=True)
    
    
# same for df_test
# 'BsmtCond' - 3.02% missing values, 44/1453
# 'BsmtQual' - 2.95% missing values, 43/1453
# 'BsmtFinType1' - 2.82% missing values, 41/1453
# 'BsmtFinType2' - 2.82% missing values, 41/1453
# 41 of them are the same (without basement)
df_test[(df_test['isBasement']==0) & (df_test['BsmtFinType1'].isnull()==True) & (df_test['BsmtFinType2'].isnull()==True) & (df_test['BsmtCond'].isnull()==True) & (df_test['BsmtQual'].isnull()==True)] # 41
# assign 'no basement' to them
columns_to_fill = ['BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']
for col in columns_to_fill:
    df_test.loc[(df_test['isBasement']==0) & ((df_test['BsmtFinType1'].isnull()==True) | (df_test['BsmtFinType2'].isnull()==True) | (df_test['BsmtCond'].isnull()==True) | (df_test['BsmtQual'].isnull()==True)), col] = 'NA'

# other unobvious rows append
unobv_basement_nulls = df_test[(df_test['BsmtCond'].isnull()==True) | (df_test['BsmtQual'].isnull()==True) | (df_test['BsmtFinType1'].isnull()==True) | (df_test['BsmtFinType2'].isnull()==True)] # 5
df_test_nulls_for_iterative_imputer = pd.concat([unobv_basement_nulls, df_test_nulls_for_iterative_imputer])
df_test = df_test.drop(unobv_basement_nulls.index)


# 'MasVnrArea' - 0.45% missing values, 6 / 1307
# MasVnrArea has a normal distribution, so we can replace nulls with mean,
# but first we'll check absent masonries
df.loc[(df['isMasonry']==0) & (df['MasVnrArea'].isnull()==True), 'MasVnrArea'] = 0 # 6

# same for df_test
# 'MasVnrArea' - 1.03% missing values, 15 / 1448
df_test.loc[(df_test['isMasonry']==0) & (df_test['MasVnrArea'].isnull()==True), 'MasVnrArea'] = 0 # 15


# 'Electrical' - 0.07% missing values, 1 / 1307
# unobvious
unobv_electrical = df[df['Electrical'].isnull()==True]
df_nulls_for_iterative_imputer = pd.concat([unobv_electrical, df_nulls_for_iterative_imputer])
df.drop(unobv_electrical.index, inplace=True)


# MSZoning - 0.27% missing values, (4/1448)
# Utilities - 0.13% missing values, (2/1448)
# Functional - 0.06% missing values, (2/1448)
# Exterior1st, Exterior2nd - 0.06% missing values, (1/1448) - same row
# KitchenQual - 0.06% missing values, (1/1448)
# SaleType - 0.06% missing values, (1/1448)		
unobvious_null_columns = ['MSZoning', 'Utilities', 'Functional', 'Exterior1st', 'KitchenQual', 'SaleType']
for col in unobvious_null_columns:
    df_test_nulls_for_iterative_imputer = pd.concat([df_test_nulls_for_iterative_imputer, df_test[df_test[col].isnull()==True]])
    df_test.drop(df_test[df_test[col].isnull()==True].index, inplace=True)
    

df.to_csv('Post_nulls_datasets/train.csv', index=False)
df_test.to_csv('Post_nulls_datasets/test.csv', index=False)
df_nulls_for_iterative_imputer.to_csv('Post_nulls_datasets/train_iterative_imputer_nulls.csv', index=False)
df_test_nulls_for_iterative_imputer.to_csv('Post_nulls_datasets/test_iterative_imputer_nulls.csv', index=False)

