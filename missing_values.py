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

#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer

# Assuming X_train is your training data with non-null columns
# and X_test is your testing data with null columns

# Create an instance of IterativeImputer
#imputer = IterativeImputer()

# Fit the imputer on the training data
#imputer.fit(X_train)

# Impute the null values in the testing data
#X_test_imputed = imputer.transform(X_test)

# X_test_imputed now contains the testing data with imputed values



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


# MasVnrType - 61.36% missing values, 802/1307
# 'None' should be 'no masonry', apply this
df.loc[(df['isMasonry']==0) & (df['MasVnrType'].isnull()==True), 'MasVnrType'] = 'None' # 798
# others are not obvious
df_nulls_for_iterative_imputer = df[df['MasVnrType'].isnull()==True]
df.drop([563,694,1166,1198], inplace=True)

# same for df_test
# MasVnrType - 61.32% missing values, 891/1453
df_test.loc[(df_test['isMasonry']==0) & (df_test['MasVnrType'].isnull()==True), 'MasVnrType'] = 'None' # 888
df_test_nulls_for_iterative_imputer = df_test.loc[[209, 992, 1150]]
df_test.drop([209,992,1150], inplace=True)





df_nulls = get_nulls(df)
df_test_nulls = get_nulls(df_test)





# 'FireplaceQu' - 50.38% missing values, 657 / 1304	
# according to the docs 'NA' should be no fireplace
# we will assign 'NA' to absent fireplaces
df[(df['isFireplace']==0) & (df['FireplaceQu'].isnull() == True)] # 657
df.loc[(df['isFireplace']==0) & (df['FireplaceQu'].isnull() == True), 'FireplaceQu'] = 'NA' # 657
df_test.loc[(df_test['isFireplace']==0) & (df_test['FireplaceQu'].isnull() == True), 'FireplaceQu'] = 'NA' # 728


# 'LotFrontage' - 17.71% missing values, 231 / 1304	
# a continuous variable with a normal distribution, so we can replace nulls with mean
imputer_mean = SimpleImputer(strategy='mean')
df['LotFrontage'] = imputer_mean.fit_transform(df[['LotFrontage']])
df_test['LotFrontage'] = imputer_mean.fit_transform(df_test[['LotFrontage']])



# 'GarageType' - 5.67% missing values, 74 / 1304
# 'GarageYrBlt' - 5.67% missing values, 74 / 1304
# 'GarageFinish' - 5.67% missing values, 74 / 1304	
# 'GarageQual' - 5.67% missing values, 74 / 1304	
# 'GarageCond' - 5.67% missing values, 74 / 1304
# --- 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond' all
# have the exact same amount of null values (5.67% missing values, 74 / 1304) +
# documentation states about 'NA' being  no garage and it's the only value missing.
# Thus, we can imply all of them are values for "no garage"; let's check that
df[(df['GarageArea']==0) & (df['GarageType'].isnull()==True) & (df['GarageYrBlt'].isnull()==True) & (df['GarageFinish'].isnull()==True) & (df['GarageQual'].isnull()==True) & (df['GarageCond'].isnull()==True)] # 74
# proved
df['GarageYrBlt'].fillna(0, inplace=True)
columns_to_fill = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in columns_to_fill:
    df[col].fillna("NA", inplace=True)

# same for df_test
# GarageFinish - 5.35% missing valus, 78/1456	
# GarageYrBlt - 5.35% missing valus, 78/1456	
# GarageCond - 5.35% missing valus, 78/1456
# GarageQual - 5.35% missing valus, 78/1456
# GarageType - 5.21% missing valus, 76/1456
df_test[(df_test['GarageArea']==0)] # 76
df_test[(df_test['GarageArea']==0) & (df_test['GarageType'].isnull()==True) & (df_test['GarageYrBlt'].isnull()==True) & (df_test['GarageFinish'].isnull()==True) & (df_test['GarageQual'].isnull()==True) & (df_test['GarageCond'].isnull()==True)] # 76
df_test['GarageYrBlt'].fillna(0, inplace=True)
for col in columns_to_fill:
    df_test[col].fillna("NA", inplace=True)

# make a binary column for easier classification
dfs_to_impute = [df, df_test, df_nulls_for_iterative_imputer, df_test_nulls_for_iterative_imputer]
for data_frame in dfs_to_impute:
    data_frame['isGarage'] = data_frame['GarageType'].apply(lambda x: 0 if x=='NA' else 1)
    

# 'BsmtExposure' - 	2.83% missing values, 37 / 1304	
# according to the docs 'NA' should be no basement, we will check if there any others
# create a column to characterize a basement by its presence or absence
df['isBasement'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df.loc[(df['isBasement']==0) & (df['BsmtExposure'].isnull()==True), 'BsmtExposure'] = 'NA' # 36
# unclassified ones add to the df_nulls_for_iterative_imputer
df_nulls_for_iterative_imputer = df_nulls_for_iterative_imputer.append(df[df['BsmtExposure'].isnull()==True], ignore_index=True)
df = df.drop([849])

# same for df_test
# BsmtExposure - 2.95% missing values, 43 / 1456	
# the TotalBsmtSF has 1 null in df_test, all basement columns are nan there
# so we cannot say if it has basement or not, not obvious
df_test['isBasement'] = np.nan # assigning nans to all isBasement rows
# appending the row with nan basement parameters to IterativeImputer
df_test_nulls_for_iterative_imputer = df_test_nulls_for_iterative_imputer.append(df_test[df_test['TotalBsmtSF'].isnull()==True], ignore_index=True)
# remove it from the test dataset
df_test = df_test.drop([660])
# setting other values of isBasement after removing TotalBsmtSF nan
df_test['isBasement'] = df_test['TotalBsmtSF'].dropna().apply(lambda x: 1 if x > 0 else 0)
df_test.loc[(df_test['isBasement']==0) & (df_test['BsmtExposure'].isnull()==True), 'BsmtExposure'] = 'NA' # 40
# others are not obvious
unobv_bsmt_exposure_nulls = df_test[df_test['BsmtExposure'].isnull()==True] # 2
df_test_nulls_for_iterative_imputer = df_test_nulls_for_iterative_imputer.append(unobv_bsmt_exposure_nulls, ignore_index=True)
df_test = df_test.drop(unobv_bsmt_exposure_nulls.index)


# 'BsmtFinType1' - 2.76% missing values, 36 / 1304	
# 'BsmtFinType2' - 2.76% missing values, 36 / 1304
# 'BsmtCond' - 2.76% missing values, 36 / 1304		
# 'BsmtQual' - 2.76% missing values, 36 / 1304	
# --- now, the 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual' all has the exact
# same amount of null values. So we can imply all of them are values for "no basement"
# let's check that
df[(df['isBasement']==0) & (df['BsmtFinType1'].isnull()==True) & (df['BsmtFinType2'].isnull()==True) & (df['BsmtCond'].isnull()==True) & (df['BsmtQual'].isnull()==True)] # 36
# proved
columns_to_fill = ['BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual']
for col in columns_to_fill:
    df[col].fillna("NA", inplace=True)
    
    
# same for df_test
# 'BsmtCond' - 3.02% missing values, 44/1456
# 'BsmtQual' - 2.95% missing values, 43/1456
# 'BsmtFinType1' - 2.81% missing values, 41/1456
# 'BsmtFinType2' - 2.81% missing values, 41/1456
# 40 of them are the same (without basement)
df_test[(df_test['isBasement']==0) & (df_test['BsmtFinType1'].isnull()==True) & (df_test['BsmtFinType2'].isnull()==True) & (df_test['BsmtCond'].isnull()==True) & (df_test['BsmtQual'].isnull()==True)] # 40
# assign 'no basement' to them
columns_to_fill = ['BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']
for col in columns_to_fill:
    df_test.loc[(df_test['isBasement']==0) & ((df_test['BsmtFinType1'].isnull()==True) | (df_test['BsmtFinType2'].isnull()==True) | (df_test['BsmtCond'].isnull()==True) | (df_test['BsmtQual'].isnull()==True)), col] = 'NA'

# other unobvious rows append
unobv_basement_nulls = df_test[(df_test['BsmtCond'].isnull()==True) | (df_test['BsmtQual'].isnull()==True) | (df_test['BsmtFinType1'].isnull()==True) | (df_test['BsmtFinType2'].isnull()==True)] # 5
df_test_nulls_for_iterative_imputer = df_test_nulls_for_iterative_imputer.append(unobv_basement_nulls, ignore_index=True)
df_test = df_test.drop(unobv_basement_nulls.index)





df_nulls = get_nulls(df)
df_test_nulls = get_nulls(df_test)





# 'MasVnrArea' - 0.46% missing values, 6 / 1303
# MasVnrArea has a normal distribution, so we can replace nulls with mean,
# but first we'll check absent masonries
df.loc[(df['isMasonry']==0) & (df['MasVnrArea'].isnull()==True), 'MasVnrArea'] = 0 # 6

# same for df_test
# 'MasVnrArea' - 1.03% missing values, 15 / 1450
df_test.loc[(df_test['isMasonry']==0) & (df_test['MasVnrArea'].isnull()==True), 'MasVnrArea'] = 0 # 15


# 'Electrical' - 0.07% missing values, 1 / 1303
# unobvious
df_nulls_for_iterative_imputer = df_nulls_for_iterative_imputer.append(df[df['Electrical'].isnull()==True], ignore_index=True)
df.drop([1236], inplace=True)


# MSZoning - 0.27% missing values, (4/1451)
# Utilities - 0.13% missing values, (1/1444)
# Functional - 0.06% missing values, (1/1443)
# Exterior1st, Exterior2nd - 0.06% missing values, (1/1442) - same row
# KitchenQual - 0.06% missing values, (1/1441)
# SaleType - 0.06% missing values, (1/1440)	
# GarageCars, GarageArea - 0.06% missing values, (1/1439) - same row	
unobvious_null_column_names = ['MSZoning', 'Utilities', 'Functional', 'Exterior1st', 'KitchenQual', 'SaleType', 'GarageCars']
for col in unobvious_null_column_names:
    df_test_nulls_for_iterative_imputer = df_test_nulls_for_iterative_imputer.append(df_test[df_test[col].isnull()==True], ignore_index=True)
    df_test.drop(df_test[df_test[col].isnull()==True].index, inplace=True)
    

df.to_csv('Post_nulls_datasets/train.csv', index=False)
df_test.to_csv('Post_nulls_datasets/test.csv', index=False)
df_nulls_for_iterative_imputer.to_csv('Post_nulls_datasets/train_iterative_imputer_nulls.csv', index=False)
df_test_nulls_for_iterative_imputer.to_csv('Post_nulls_datasets/test_iterative_imputer_nulls.csv', index=False)

