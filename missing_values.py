import pandas as pd
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
df_nulls_test = get_nulls(df_test)


# 'PoolQC' - more than 99% missing data, only 2 non-null values, exclude it
df.drop('PoolQC', axis=1, inplace=True)
df_test.drop('PoolQC', axis=1, inplace=True)

# 'Alley' - 93.9% missing values, 1229/1308
# according to the docs 'NA' should be no alley access, so we might replace nulls with NA
# but it is extremely imbalanced and unlikely to help in predicting, exclude it
df.drop('Alley', axis=1, inplace=True)
df_test.drop('Alley', axis=1, inplace=True)

# 'Fence' - 80.19% missing values, 1049 / 1308	
# according to the docs 'NA' should be no fence, so we will replace nulls with NA
df['Fence'].fillna("NA", inplace=True)
df_test['Fence'].fillna("NA", inplace=True)

# 'MasVnrType' - 61.31% missing values, 802 / 1308	
# according to the docs 'NA' should be no masonry, but there are also another values absent
# in the column, so we will check, is any isMasonry == 1 (had been created before to classify
# absense or presence of a masonry) has a null in MasVnrType?
df[df['isMasonry']==1]['MasVnrType'].isnull().sum() # 4
df_test[df_test['isMasonry']==1]['MasVnrType'].isnull().sum() # 3
# it has 4 nulls, so it is better to drop those, replace nulls with "None" and it will
# signify that there's no masonry there
masonry_nulls_df = df[df['isMasonry']==1]['MasVnrType'].isnull()
masonry_nulls_df = masonry_nulls_df[masonry_nulls_df==True]
df.drop(masonry_nulls_df.index, inplace=True)
df['MasVnrType'].fillna("None", inplace=True)

masonry_nulls_df_test = df_test[df_test['isMasonry']==1]['MasVnrType'].isnull()
masonry_nulls_df_test = masonry_nulls_df_test[masonry_nulls_df_test==True]
df_test.drop(masonry_nulls_df_test.index, inplace=True)
df_test['MasVnrType'].fillna("None", inplace=True)


df_nulls = get_nulls(df)
# 'FireplaceQu' - 50.38% missing values, 657 / 1304	
# according to the docs 'NA' should be no fireplace, so we will replace them with 'NA'
df['FireplaceQu'].fillna("NA", inplace=True)
df_test['FireplaceQu'].fillna("NA", inplace=True)

# 'LotFrontage' - 17.71% missing values, 231 / 1304	
# a continuous variable with a normal distribution, so we can replace nulls with mean
imputer_mean = SimpleImputer(strategy='mean')
df['LotFrontage'] = imputer_mean.fit_transform(df[['LotFrontage']])
df_test['LotFrontage'] = imputer_mean.fit_transform(df_test[['LotFrontage']])


df_nulls = get_nulls(df)
df_nulls_test = get_nulls(df_test)


# --- 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond' all
# have the exact same amount of null values (5.67% missing values, 74 / 1304) +
# documentation states about 'NA' being  no garage and it's the only value missing.
# Thus, we can imply all of them are values for "no garage"; let's check that
df['GarageArea'].isnull().any() # False
# use GarageArea as a determiner whether there is a garage or not
no_garage_hypothesis_df = df[df['GarageArea']==0] # 74
# 'GarageType' - 5.67% missing values, 74 / 1304	
no_garage_hypothesis_df.index.equals(df[df['GarageType'].isnull()==True].index) # True
# 'GarageYrBlt' - 5.67% missing values, 74 / 1304
no_garage_hypothesis_df.index.equals(df[df['GarageYrBlt'].isnull()==True].index) # True
# 'GarageFinish' - 5.67% missing values, 74 / 1304	
no_garage_hypothesis_df.index.equals(df[df['GarageFinish'].isnull()==True].index) # True
# 'GarageQual' - 5.67% missing values, 74 / 1304	
no_garage_hypothesis_df.index.equals(df[df['GarageQual'].isnull()==True].index) # True
# 'GarageCond' - 5.67% missing values, 74 / 1304	
no_garage_hypothesis_df.index.equals(df[df['GarageCond'].isnull()==True].index) # True

# proved
df['GarageYrBlt'].fillna(0, inplace=True)
columns_to_fill = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in columns_to_fill:
    df[col].fillna("NA", inplace=True)

    
df_nulls = get_nulls(df)
df_nulls_test = get_nulls(df_test)



# 'BsmtExposure' - 	2.83% missing values, 37 / 1304	
# according to the docs 'NA' should be no basement, we will check if there any others
df[df['TotalBsmtSF']==0].shape[0] # 36 rows
df_test[df_test['TotalBsmtSF']==0].shape[0] # 40 rows
# only they can be considered 'no basement'
df.loc[df[df['TotalBsmtSF']==0].index, 'BsmtExposure'] = df.loc[df[df['TotalBsmtSF']==0].index, 'BsmtExposure'].fillna("NA")
df_test.loc[df_test[df_test['TotalBsmtSF']==0].index, 'BsmtExposure'] = df_test.loc[df_test[df_test['TotalBsmtSF']==0].index, 'BsmtExposure'].fillna("NA")
# remove the one extra
df.drop(df[df['BsmtExposure'].isnull()==True].index, inplace=True)
df_test.drop(df_test[df_test['BsmtExposure'].isnull()==True].index, inplace=True)

# --- now, the 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual' all has the exact
# same amount of null values. So we can imply all of them are values for "no basement"
# let's check that
no_basemet_hypothesis_df = df[df['TotalBsmtSF']==0] # 4
# 'BsmtFinType1' - 2.76% missing values, 36 / 1304	
no_basemet_hypothesis_df.index.equals(df[df['BsmtFinType1'].isnull()==True].index) # True
# 'BsmtFinType2' - 2.76% missing values, 36 / 1304	
no_basemet_hypothesis_df.index.equals(df[df['BsmtFinType2'].isnull()==True].index) # True
# 'BsmtCond' - 2.76% missing values, 36 / 1304	
no_basemet_hypothesis_df.index.equals(df[df['BsmtCond'].isnull()==True].index) # True
# 'BsmtQual' - 2.76% missing values, 36 / 1304	
no_basemet_hypothesis_df.index.equals(df[df['BsmtQual'].isnull()==True].index) # True

# same for df_test
no_basement_hypothesis_df_test = df_test[df_test['TotalBsmtSF']==0] # 3
no_basement_hypothesis_df_test.index.equals(df_test[df_test['BsmtFinType1'].isnull()==True].index) # True
no_basement_hypothesis_df_test.index.equals(df_test[df_test['BsmtFinType2'].isnull()==True].index) # True
no_basement_hypothesis_df_test.index.equals(df_test[df_test['BsmtCond'].isnull()==True].index) # True
no_basement_hypothesis_df_test.index.equals(df_test[df_test['BsmtQual'].isnull()==True].index) # True
# proved
columns_to_fill = ['BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual']
for col in columns_to_fill:
    df[col].fillna("NA", inplace=True)
    df_test[col].fillna("NA", inplace=True)


# 'MasVnrArea' - 0.46% missing values, 6 / 1303
df[df['MasVnrArea'].isnull()==True]['isMasonry']
# according to the 'isMasonery' value, all of the nulls doesn't have a masonry, so the sf = 0
df.loc[df['MasVnrArea'].isnull(), 'MasVnrArea'] = 0
# same is applicable to the df_test data frame:
# 'MasVnrArea' - 1.03% missing values, 15 / 1453
df_test[df_test['MasVnrArea'].isnull()==True]['isMasonry']
df_test.loc[df_test['MasVnrArea'].isnull(), 'MasVnrArea'] = 0

# 'Electrical' - 0.07% missing values, 1 / 1303
df[df['Electrical'].isnull()==True].T
# it is better to drop 1 row, than make a guess
df.drop(df[df['Electrical'].isnull()].index, inplace=True)


df_nulls = get_nulls(df) # no more nulls here
df_nulls_test = get_nulls(df_test) # 14 more columns here


# GarageYrBlt - 5.36% (78/1453)
# GarageFinish - 5.36% (78/1453)
# GarageQual - 5.36% (78/1453)
# GarageCond - 5.36% (78/1453)
# GarageType - 5.23% (76/1453)
# many similar values, let us handle them all together
# first, we will check if those are the same 78 rows
GarageYrBlt_df_test_nulls = df_test[df_test['GarageYrBlt'].isnull()==True].index
GarageFinish_df_test_nulls = df_test[df_test['GarageFinish'].isnull()==True].index
GarageQual_df_test_nulls = df_test[df_test['GarageQual'].isnull()==True].index
GarageCond_df_test_nulls = df_test[df_test['GarageCond'].isnull()==True].index
all(indexes.equals(GarageYrBlt_df_test_nulls) for indexes in [GarageFinish_df_test_nulls, GarageQual_df_test_nulls, GarageCond_df_test_nulls]) # True
# they are the same. Now, we will check wether the GarageType's 76 rows are in those 78
GarageType_df_test_nulls = df_test[df_test['GarageType'].isnull()==True].index
all(GarageType_df_test_nulls.isin(GarageYrBlt_df_test_nulls)) # True
# they are. Now, get those 2 rows of difference to handle them separately first
GarageType_df_test_nulls.symmetric_difference(GarageYrBlt_df_test_nulls) # 666, 1116
# both rows doesn't have a significant amount of information about garages.
# in fact, almost all of it is missing. As we are not allowed to drop anything 
# in a test data frame, we will save these separately to further predict values
# with the help of a model that will not consider garages parameters 
# (only its presence (1) or absense (0) )
df['isGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
# symbolically assign 1 to mark that garage is there. We can claim so because
# both rows has the GarageType specified as Detached (not null or NA).
df_test.loc[[1116], 'GarageArea'] = 1 # only for 1116 because 666 has GarageArea specified
df_test['isGarage'] = df_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
null_garages_df_test = df_test.loc[[666, 1116]] # rows for the separate model
null_garages_df_test.drop(['GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond'], inplace=True, axis=1)
null_garages_df_test.to_csv("Post_nulls_datasets/null_garages.csv", index=False)
df_test.drop([666, 1116], inplace=True)
# now we can look at 76 same rows with a lot of nulls regarding the garages
tmp = df_test[df_test['GarageYrBlt'].isnull()==True].T
# as all garage cars and garage areas equals 0, we can presume there are no garages
# so we will replce nulls accorind to the docs
df_test['GarageYrBlt'].fillna(0, inplace=True)
columns_to_fill = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in columns_to_fill:
    df_test[col].fillna("NA", inplace=True)


df_nulls_test = get_nulls(df_test)



# MSZoning - 0.27% (4/1453)	
# Utilities - 0.13% (2/1453)
# Functional - 0.13% (2/1453)
# Exterior1st - 0.06% (1/1453)	
# Exterior2nd - 0.06% (1/1453)
# KitchenQual - 0.06% (1/1453
# KitchenQual - 0.06% (1/1453)
# GarageCars - 0.06% (1/1453)
# GarageArea - 0.06% (1/1453)
# SaleType - 0.06% (1/1453)	




# 


















