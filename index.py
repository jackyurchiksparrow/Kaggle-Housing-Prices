import pandas as pd
#import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv("train.csv", index_col="Id")

# basic data description
description_numerical = df.select_dtypes(include='number').describe().T
description_categorical = df.select_dtypes(include='object').describe().T

# inspect the null values
def get_nulls(df):
    null_columns = df.isnull().sum()
    null_columns = null_columns[null_columns>0]
    df_nulls = pd.DataFrame({"null_count": null_columns, "null_percent": (null_columns/df.shape[0])*100}).sort_values(by='null_count', ascending=False)
    print(df_nulls)
    return df_nulls

df_nulls = get_nulls(df)

# research them
print(df['PoolQC'].unique()) # make ordinal

def PoolQC_lab_encoding(series):
    if series == 'Ex': # excellent pool
        return 4
    elif series == 'Gd': # good pool
        return 3
    elif series == 'TA': # average pool
        return 2
    elif series == 'Fa': # fair pool
        return 1
    else:
        return 0 # no pool
    
df['PoolQC'] = df['PoolQC'].apply(lambda x: PoolQC_lab_encoding(x))

df_nulls = get_nulls(df)

print(df['MiscFeature'].unique()) 
df['Shed'] = df['MiscFeature'].apply(lambda x: 1 if x=="Shed" else 0)

print(df[df['MiscFeature']=='Gar2']['GarageType'])
# so it's not specified in 'GarageType', although we do have 'More than one type of garage'
# option in 'GarageType', therefore we will make 'GarageType' nominal and include both 
# and drop 'GarageType'
df['GarageType'] = df['GarageType'].fillna("NA")
df_nulls = get_nulls(df)
# add it to preprocessing object and then
df['MiscFeature'] = df['MiscFeature'].fillna("NA")
df['MiscFeature'] = df['MiscFeature'].apply(lambda x: "NA" if x=="Gar2" else x)

df_nulls = get_nulls(df)

# nominal alley
print(df['Alley'].unique())
df['Alley'] = df['Alley'].fillna("NA")
# then include to preprocessing and drop

df_nulls = get_nulls(df)

# nominal fence
print(df['Fence'].unique())
df['Fence'] = df['Fence'].fillna("NA")
# then include to preprocessing and drop

df_nulls = get_nulls(df)
# nominal MasVnrType
print(df['MasVnrType'].unique())
df['MasVnrType'] = df['MasVnrType'].fillna("None")
# then include to preprocessing and drop

df_nulls = get_nulls(df)

# ordinal FireplaceQu
print(df['FireplaceQu'].unique())
df['FireplaceQu'] = df['FireplaceQu'].fillna("NA")

def dFireplaceQu_lab_encoding(series):
    if series == 'Ex': # excellent pool
        return 5
    elif series == 'Gd': # good pool
        return 4
    elif series == 'TA': # average pool
        return 3
    elif series == 'Fa': # fair pool
        return 2
    elif series == 'Po':
        return 1
    else:
        return 0 # no pool

df['FireplaceQu'] = df['FireplaceQu'].apply(lambda x: dFireplaceQu_lab_encoding(x))
    
df_nulls = get_nulls(df)

# numerical col with nans
print(df['LotFrontage'].unique())
print(df['LotFrontage'].value_counts())
LotFrontage_values = df['LotFrontage'].value_counts()
# make them ordinal with the largest values [75; 143]
def LotFrontage_lab_encoding(series):
    # <75, 75, <80, 80, <85
    if series < 75:
        return 0
    elif series == 75:
        return 1
    elif series < 80:
        return 2
    elif series == 80:
        return 3
    else:
        return 4
        

df['LotFrontage'] = df['LotFrontage'].apply(lambda x: LotFrontage_lab_encoding(x))

df_nulls = get_nulls(df)

print(df['GarageYrBlt'].unique())
print(df['GarageYrBlt'].value_counts())
GarageYrBlt_values = df['GarageYrBlt'].value_counts()
# make them ordinal with the largest values [2003; 2006]
# <2003, 2003, 2004, 2005, 2006, 2007, > 2007
def GarageYrBlt_lab_encoding(series):
    if series < 2003:
        return 0
    elif series == 2003:
        return 1
    elif series == 2004:
        return 2
    elif series == 2005:
        return 3
    elif series == 2006:
        return 4
    elif series == 2007:
        return 5
    else:
        return 6
    
df['GarageYrBlt'] = df['GarageYrBlt'].apply(lambda x: GarageYrBlt_lab_encoding(x))

df_nulls = get_nulls(df)

print(df['GarageFinish'].unique())
print(df['GarageFinish'].value_counts())
df['GarageFinish'] = df['GarageFinish'].fillna("NA")
# it's nominal
# then include to preprocessing and drop

df_nulls = get_nulls(df)

# one hot encode categorical features
ohe = OneHotEncoder(handle_unknown='ignore')
# store one hot encoder in a pipeline
categorical_processing = Pipeline(steps=[('ohe', ohe)])
# create the ColumnTransormer object
# remainder parameter ensures that the columns not specified in the 
# transformer are not dropped.
preprocessing = ColumnTransformer(transformers=[('categorical', categorical_processing, 
['GarageType', 'Alley', 'Fence', 'MasVnrType', 'GarageFinish'])], remainder='passthrough')
df.drop(['GatageType', 'Alley', 'Fence', 'MasVnrType', 'GarageFinish'])

# check for categorical values in the dataset
#categorical_cols = df.select_dtypes(include='object').keys()
#print(f'Categorical columns {len(categorical_cols)}:')
#print(categorical_cols)

# for each categorical row, define if it's ordinal (with ordered series: marks, blood group) 
# or nominal (no ordering or ranking system: gender, race)
# we will apply one hot encoding for nominal and label encoding for ordinal
# we should also pay attention if the categorical row has a high cardinality
# then we will reduce dimensionality by generalizing the data

# 1. drop MSSubClass?
# 2. MSZoning - nominal, leave categories Agriculture, Commercial, Residential, Industrial
# 3. street - nominal
# 4. alley - nominal
# 5. LotShape - nominal, leave regular/irregular only
# 6. LandContour - nominal
# 7. utilities - make ordinal count 4 3 2 1
# 8. LotConfig - nominal
# 9. 




# one hot encode categorical features
#ohe = OneHotEncoder(handle_unknown='ignore')

# store one hot encoder in a pipeline
#categorical_processing = Pipeline(steps=[('ohe', ohe)])

# create the ColumnTransormer object
# remainder parameter ensures that the columns not specified in the 
# transformer are not dropped.
#preprocessing = ColumnTransformer(transformers=[('categorical', categorical_processing, 
#['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
#'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
# '', '','', '', '', '', '', '', '', '', '', '', '', '', ''])], remainder='passthrough')

#data science how to normalize years

#analyze all nans (lot fontage turn nans to 0)

# turn mszoning, street, alley, lotshape, 
# LandContour, Utilities, LotConfig,
# LandSlope, Neighborhood, Condition1,
# Condition2, BldgType, HouseStyle, 
# RoofStyle, RoofMatl, Exterior1st, 
# Exterior2nd, MasVnrType, MasVnrArea, 
# ExterQual, ExterCond, Foundation, 
# BsmtQual, BsmtCond, BsmtExposure, 
# BsmtFinType1, BsmtFinType2,
# Heating, HeatingQC, CentralAir, 
# Electrical, KitchenQual, ?Functional?,
# FireplaceQu, GarageType, GarageFinish
# GarageQual, GarageCond, PavedDrive, 
# PoolQC, Fence, MiscFeature,
# SaleType, SaleCondition into numerical values

#condition2 is same as condition1? if no remodeling or additions
#we should make it as an additional boolean col


#BsmtFinType2 is same as BsmtFinType1? if no remodeling or additions
#we should make it as an additional boolean col
# nulls should be replaced with mode

#Exterior2nd is same as Exterior1st? if no remodeling or additions
#we should make it as an additional boolean col

#yearbuilt, yearremodadd - if there are too
# much different years classify years
#into categories and 0 or 1 if it was remastered

#yearremodadd is same as yearbuilt if no remodeling or additions
#we should make it as an additional boolean col


# identify outliers



import matplotlib.pyplot as plt

data = [1, 2, 3, 4, 5, 20]

plt.boxplot(data)
plt.show()




