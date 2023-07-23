import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer

# na_filter=False because it understood 'NA' as nan
df = pd.read_csv("Post_nulls_datasets/train.csv", na_filter=False)
df_test = pd.read_csv("Post_nulls_datasets/test.csv", na_filter=False)
df_nulls_for_iterative_imputer = pd.read_csv("Post_nulls_datasets/train_iterative_imputer_nulls.csv", na_filter=False)
df_test_nulls_for_iterative_imputer = pd.read_csv("Post_nulls_datasets/test_iterative_imputer_nulls.csv", na_filter=False)

# make empty cells nans and not strings
df_nulls_for_iterative_imputer.replace('', np.nan, inplace=True)
df_test_nulls_for_iterative_imputer.replace('', np.nan, inplace=True)


# 1. Determine which columns require categorical processing
# all non-numbers for sure
categorical_cols = df.select_dtypes(include='object')
numeric_cols = df.select_dtypes(include='number')

# first, we will research ambigious fields, such as year
# if we have many years and only a few are used a lot, we would
# want to perform data binning
sns.histplot(data=df, x='YearBuilt')
sns.histplot(data=df, x='YearRemodAdd')
sns.histplot(data=df[df['GarageYrBlt']>0], x='GarageYrBlt')
sns.histplot(data=df, x='Neighborhood')
# none of the is the case

def One_Hot_Encode_column(encoder, column, data_frame):
    transformed = encoder.transform(data_frame[column].values.reshape(-1, 1))  # Reshape to a 2D array
    feature_names = encoder.get_feature_names_out([column])
    transformed_df = pd.DataFrame(transformed.toarray(), columns=feature_names)
    data_frame.drop(column, inplace=True, axis=1)  # Drop the original column inplace
    data_frame[feature_names] = transformed_df   # Join the transformed one-hot encoded DataFrame
    return data_frame



def ordinal_encode_column(dictionary_values, column, data_frames):
    for data_frame in data_frames:
        data_frame[column] = data_frame[column].apply(lambda x: values[x] if x in dictionary_values else x)



nominals = ['MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 'BldgType',
           'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'Electrical',
           'Functional', 'GarageType', 'SaleType', 'SaleCondition', 'MSSubClass']

nominal_multilabeled = ['Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd', 
                        'BsmtFinType1', 'BsmtFinType2']

ordinals = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual', 'BsmtQual',
            'BsmtCond', 'FireplaceQu', 'GarageQual', 'GarageCond', 'BsmtExposure', 'CentralAir',
            'GarageFinish', 'PavedDrive', 'Utilities']

all_dfs = [df,df_test, df_nulls_for_iterative_imputer, df_test_nulls_for_iterative_imputer]

train_columns_with_nulls = df_nulls_for_iterative_imputer.columns[df_nulls_for_iterative_imputer.isnull().any()].tolist()
test_columns_with_nulls = df_test_nulls_for_iterative_imputer.columns[df_test_nulls_for_iterative_imputer.isnull().any()].tolist()

# ----------------------- dealing with ordinal columns first ----------------------

# LotShape - ordinal data; signifies irregularity of shape (bigger number more irregular shape)
values = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
ordinal_encode_column(values, 'LotShape', all_dfs)

# LandSlope - ordinal (bigger number - biger slope)
values = {'Gtl': 1, 'Mod': 2, 'Sev': 3}
ordinal_encode_column(values, 'LandSlope', all_dfs)

# ExterQual - ordinal (bigger number - better quality)
values = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ordinal_encode_column(values, 'ExterQual', all_dfs)
# ExterCond - ordinal (bigger number - better quality)
ordinal_encode_column(values, 'ExterCond', all_dfs)
# HeatingQC - ordinal (bigger number - better quality)
ordinal_encode_column(values, 'HeatingQC', all_dfs)
# KitchenQual - ordinal (bigger number - better quality)
ordinal_encode_column(values, 'KitchenQual', all_dfs)
# BsmtQual - ordinal (bigger number - better quality)
values = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ordinal_encode_column(values, 'BsmtQual', all_dfs)
# BsmtCond - ordinal (bigger number - better condition)
ordinal_encode_column(values, 'BsmtCond', all_dfs)
# FireplaceQu - ordinal (bigger number - better quality)
ordinal_encode_column(values, 'FireplaceQu', all_dfs)
# GarageQual - ordinal (bigger number - better quality)
ordinal_encode_column(values, 'GarageQual', all_dfs)
# GarageCond - ordinal (bigger number - better quality)
ordinal_encode_column(values, 'GarageCond', all_dfs)

# BsmtExposure - ordinal (bigger number - better exposure)
values = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
ordinal_encode_column(values, 'BsmtExposure', all_dfs)
# CentralAir - ordinal binary
values = {'Y': 1, 'N': 0}
ordinal_encode_column(values, 'CentralAir', all_dfs)
# GarageFinish - ordinal (bigger number - closer to finish)
values = {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
ordinal_encode_column(values, 'GarageFinish', all_dfs)
# PavedDrive - ordinal (bigger number - better pavement)
values = {'Y': 2, 'P': 1, 'N': 0}
ordinal_encode_column(values, 'PavedDrive', all_dfs)
# Utilities - ordinal, because it has only two values in the data frames and all
# the values are related
values = {'AllPub': 4, 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1}
ordinal_encode_column(values, 'Utilities', all_dfs)

# ----------------------- dealing with multilabeled ----------------------
# Condition1, Condition2 - nominal data, but needs to be multilabeled

# df
condition_dummies1 = pd.get_dummies(df['Condition1'], prefix='Condition').astype(int)
condition_dummies2 = pd.get_dummies(df['Condition2'], prefix='Condition').astype(int)

for index, row in condition_dummies2.iterrows():
    for col, value in row.items():
        if value == 1:
            condition_dummies1.loc[index, col] = 1   
            
df = pd.concat([df, condition_dummies1], axis=1)
df.drop(['Condition1', 'Condition2'], axis=1, inplace=True)


# df_test
condition_dummies1 = pd.get_dummies(df_test['Condition1'], prefix='Condition').astype(int)
condition_dummies2 = pd.get_dummies(df_test['Condition2'], prefix='Condition').astype(int)

for index, row in condition_dummies2.iterrows():
    for col, value in row.items():
        if value == 1:
            condition_dummies1.loc[index, col] = 1   
            
df_test = pd.concat([df_test, condition_dummies1], axis=1)
df_test.drop(['Condition1', 'Condition2'], axis=1, inplace=True)


# df_nulls_for_iterative_imputer
# conditions don't have nulls
all_conditions = pd.DataFrame(0, index=[i for i in range(df_nulls_for_iterative_imputer.shape[0])], columns=condition_dummies1.columns)
condition_dummies1 = pd.get_dummies(df_nulls_for_iterative_imputer['Condition1'], prefix='Condition').astype(int)

all_conditions['Condition_Norm'] = condition_dummies1['Condition_Norm']

condition_dummies2 = pd.get_dummies(df_nulls_for_iterative_imputer['Condition2'], prefix='Condition').astype(int)
all_conditions['Condition_Norm'] = condition_dummies2['Condition_Norm']

df_nulls_for_iterative_imputer = pd.concat([df_nulls_for_iterative_imputer, all_conditions], axis=1)
df_nulls_for_iterative_imputer.drop(['Condition1', 'Condition2'], axis=1, inplace=True)


# df_test_nulls_for_iterative_imputer
# conditions don't have nulls
all_conditions = pd.DataFrame(0, index=[i for i in range(df_test_nulls_for_iterative_imputer.shape[0])], columns=all_conditions.columns)
condition_dummies1 = pd.get_dummies(df_test_nulls_for_iterative_imputer['Condition1'], prefix='Condition').astype(int)

all_conditions['Condition_Artery'] = condition_dummies1['Condition_Feedr']
all_conditions['Condition_Feedr'] = condition_dummies1['Condition_Feedr']
all_conditions['Condition_Norm'] = condition_dummies1['Condition_Norm']

condition_dummies2 = pd.get_dummies(df_test_nulls_for_iterative_imputer['Condition2'], prefix='Condition').astype(int)
all_conditions['Condition_Norm'] = condition_dummies2['Condition_Norm']

df_test_nulls_for_iterative_imputer = pd.concat([df_test_nulls_for_iterative_imputer, all_conditions], axis=1)
df_test_nulls_for_iterative_imputer.drop(['Condition1', 'Condition2'], axis=1, inplace=True)


# Exterior1st, Exterior2nd - nominal data, but needs to be multilabeled
# df 
exter_values = ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'WdShing', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock', 'Other', 'PreCast']
all_exteriors = pd.DataFrame(0, index=[i for i in range(df.shape[0])], columns=['Exterior_' + x for x in exter_values], dtype=int)

exter_dummies1 = pd.get_dummies(df['Exterior1st'], prefix='Exterior').astype(int)

for index, row in exter_dummies1.iterrows():
    for col, value in row.items():
        if value == 1:
            all_exteriors.loc[index, col] = 1 

exter_dummies2 = pd.get_dummies(df['Exterior2nd'], prefix='Exterior').astype(int)

for index, row in exter_dummies2.iterrows():
    for col, value in row.items():
        if value == 1:
            all_exteriors.loc[index, col] = 1   
            
df = pd.concat([df, all_exteriors], axis=1)
df.drop(['Exterior1st', 'Exterior2nd'], axis=1, inplace=True)


# df_test
all_exteriors = pd.DataFrame(0, index=[i for i in range(df_test.shape[0])], columns=['Exterior_' + x for x in exter_values], dtype=int)

exter_dummies1 = pd.get_dummies(df_test['Exterior1st'], prefix='Exterior').astype(int)

for index, row in exter_dummies1.iterrows():
    for col, value in row.items():
        if value == 1:
            all_exteriors.loc[index, col] = 1 

exter_dummies2 = pd.get_dummies(df_test['Exterior2nd'], prefix='Exterior').astype(int)

for index, row in exter_dummies2.iterrows():
    for col, value in row.items():
        if value == 1:
            all_exteriors.loc[index, col] = 1 
            
df_test = pd.concat([df_test, all_exteriors], axis=1)
df_test.drop(['Exterior1st', 'Exterior2nd'], axis=1, inplace=True)


# df_nulls_for_iterative_imputer
# exteriors don't have nulls
all_exteriors = pd.DataFrame(0, index=[i for i in range(df_nulls_for_iterative_imputer.shape[0])], columns=['Exterior_' + x for x in exter_values], dtype=int)

exter_dummies1 = pd.get_dummies(df_nulls_for_iterative_imputer['Exterior1st'], prefix='Exterior').astype(int)

all_exteriors['Exterior_VinylSd'] = exter_dummies1['Exterior_VinylSd']

exter_dummies2 = pd.get_dummies(df_nulls_for_iterative_imputer['Exterior2nd'], prefix='Exterior').astype(int)

all_exteriors['Exterior_VinylSd'] = exter_dummies2['Exterior_VinylSd']
            
df_nulls_for_iterative_imputer = pd.concat([df_nulls_for_iterative_imputer, all_exteriors], axis=1)
df_nulls_for_iterative_imputer.drop(['Exterior1st', 'Exterior2nd'], axis=1, inplace=True)


# df_test_nulls_for_iterative_imputer
# exteriors have nulls
all_exteriors = pd.DataFrame(0, index=[i for i in range(df_test_nulls_for_iterative_imputer.shape[0])], columns=['Exterior_' + x for x in exter_values], dtype=int)
exterior_nulls = df_test_nulls_for_iterative_imputer[df_test_nulls_for_iterative_imputer['Exterior1st'].isnull() == True].index

exter_dummies1 = pd.get_dummies(df_test_nulls_for_iterative_imputer['Exterior1st'], prefix='Exterior', dummy_na=False).astype(int)

for index, row in exter_dummies1.iterrows():
    for col, value in row.items():
        if value == 1:
            all_exteriors.loc[index, col] = 1 

exter_dummies2 = pd.get_dummies(df_test_nulls_for_iterative_imputer['Exterior2nd'], prefix='Exterior').astype(int)

for index, row in exter_dummies2.iterrows():
    for col, value in row.items():
        if value == 1:
            all_exteriors.loc[index, col] = 1 
            

all_exteriors.loc[exterior_nulls] = np.nan            
            
df_test_nulls_for_iterative_imputer = pd.concat([df_test_nulls_for_iterative_imputer, all_exteriors], axis=1)
df_test_nulls_for_iterative_imputer.drop(['Exterior1st', 'Exterior2nd'], axis=1, inplace=True)


# BsmtFinType1, BsmtFinType2 - nominal data, but needs to be multilabeled
# df 
bsmtFinType_values = ['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'NA', 'LwQ']
all_bsmtFinTypes = pd.DataFrame(0, index=[i for i in range(df.shape[0])], columns=['BsmtFinType_' + x for x in bsmtFinType_values], dtype=int)

bsmtFinType_dummies1 = pd.get_dummies(df['BsmtFinType1'], prefix='BsmtFinType').astype(int)

for index, row in bsmtFinType_dummies1.iterrows():
    for col, value in row.items():
        if value == 1:
            all_bsmtFinTypes.loc[index, col] = 1 

bsmtFinType_dummies2 = pd.get_dummies(df['BsmtFinType2'], prefix='BsmtFinType').astype(int)

for index, row in bsmtFinType_dummies2.iterrows():
    for col, value in row.items():
        if value == 1:
            all_exteriors.loc[index, col] = 1   
            
df = pd.concat([df, all_bsmtFinTypes], axis=1)
df.drop(['BsmtFinType1', 'BsmtFinType2'], axis=1, inplace=True)


# df_test
all_bsmtFinTypes = pd.DataFrame(0, index=[i for i in range(df_test.shape[0])], columns=['BsmtFinType_' + x for x in bsmtFinType_values], dtype=int)

bsmtFinType_dummies1 = pd.get_dummies(df_test['BsmtFinType1'], prefix='BsmtFinType').astype(int)

for index, row in bsmtFinType_dummies1.iterrows():
    for col, value in row.items():
        if value == 1:
            all_bsmtFinTypes.loc[index, col] = 1 

bsmtFinType_dummies2 = pd.get_dummies(df_test['BsmtFinType2'], prefix='BsmtFinType').astype(int)

for index, row in bsmtFinType_dummies2.iterrows():
    for col, value in row.items():
        if value == 1:
            all_exteriors.loc[index, col] = 1   
            
df_test = pd.concat([df_test, all_bsmtFinTypes], axis=1)
df_test.drop(['BsmtFinType1', 'BsmtFinType2'], axis=1, inplace=True)


# df_nulls_for_iterative_imputer
# bsmtFinTypes don't have nulls
all_bsmtFinTypes = pd.DataFrame(0, index=[i for i in range(df_nulls_for_iterative_imputer.shape[0])], columns=['BsmtFinType_' + x for x in bsmtFinType_values], dtype=int)

bsmtFinType_dummies1 = pd.get_dummies(df_nulls_for_iterative_imputer['BsmtFinType1'], prefix='BsmtFinType').astype(int)

all_bsmtFinTypes['BsmtFinType_Unf'] = bsmtFinType_dummies1['BsmtFinType_Unf']

bsmtFinType_dummies2 = pd.get_dummies(df_nulls_for_iterative_imputer['BsmtFinType2'], prefix='BsmtFinType').astype(int)

all_bsmtFinTypes['BsmtFinType_Unf'] = bsmtFinType_dummies2['BsmtFinType_Unf'] 
            
df_nulls_for_iterative_imputer = pd.concat([df_nulls_for_iterative_imputer, all_bsmtFinTypes], axis=1)
df_nulls_for_iterative_imputer.drop(['BsmtFinType1', 'BsmtFinType2'], axis=1, inplace=True)


# df_test_nulls_for_iterative_imputer
# bsmtFinTypes have nulls
all_bsmtFinTypes = pd.DataFrame(0, index=[i for i in range(df_test_nulls_for_iterative_imputer.shape[0])], columns=['BsmtFinType_' + x for x in bsmtFinType_values], dtype=int)
bsmtFinTypes_nulls = df_test_nulls_for_iterative_imputer[df_test_nulls_for_iterative_imputer['BsmtFinType1'].isnull() == True].index

bsmtFinType_dummies1 = pd.get_dummies(df_test_nulls_for_iterative_imputer['BsmtFinType1'], prefix='BsmtFinType', dummy_na=False).astype(int)

for index, row in bsmtFinType_dummies1.iterrows():
    for col, value in row.items():
        if value == 1:
            all_bsmtFinTypes.loc[index, col] = 1 

bsmtFinType_dummies2 = pd.get_dummies(df_test_nulls_for_iterative_imputer['BsmtFinType2'], prefix='BsmtFinType', dummy_na=False).astype(int)

for index, row in bsmtFinType_dummies2.iterrows():
    for col, value in row.items():
        if value == 1:
            all_bsmtFinTypes.loc[index, col] = 1 
            

all_bsmtFinTypes.loc[bsmtFinTypes_nulls] = np.nan            
            
df_test_nulls_for_iterative_imputer = pd.concat([df_test_nulls_for_iterative_imputer, all_bsmtFinTypes], axis=1)
df_test_nulls_for_iterative_imputer.drop(['BsmtFinType1', 'BsmtFinType2'], axis=1, inplace=True)



# ----------------------- dealing with nominals ----------------------

# first, we will find which categorical values differ in df and df_test
differences = list()
for col in nominals:
    diff = set(df[col].unique()).symmetric_difference(set(df_test[col].unique()))
    if len(diff) > 0:
        differences.append({col:set(np.concatenate([df[col].unique(), df_test[col].unique()], axis=0))})



# therefore, we will need to specify them manually (categories)    
HouseStyle_encoder = OneHotEncoder(handle_unknown="ignore", categories=[['1.5Fin','1.5Unf','1Story','2.5Fin','2.5Unf','2Story','SFoyer','SLvl']])
HouseStyle_encoder.fit(df[['HouseStyle']])

df = One_Hot_Encode_column(HouseStyle_encoder, 'HouseStyle', df)
df_test = One_Hot_Encode_column(HouseStyle_encoder, 'HouseStyle', df_test)
df_nulls_for_iterative_imputer = One_Hot_Encode_column(HouseStyle_encoder, 'HouseStyle', df_nulls_for_iterative_imputer)
df_test_nulls_for_iterative_imputer = One_Hot_Encode_column(HouseStyle_encoder, 'HouseStyle', df_test_nulls_for_iterative_imputer)


RoofMatl_encoder = OneHotEncoder(handle_unknown="ignore", categories=[['CompShg', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl']])
RoofMatl_encoder.fit(df[['RoofMatl']])

df = One_Hot_Encode_column(RoofMatl_encoder, 'RoofMatl', df)
df_test = One_Hot_Encode_column(RoofMatl_encoder, 'RoofMatl', df_test)
df_nulls_for_iterative_imputer = One_Hot_Encode_column(RoofMatl_encoder, 'RoofMatl', df_nulls_for_iterative_imputer)
df_test_nulls_for_iterative_imputer = One_Hot_Encode_column(RoofMatl_encoder, 'RoofMatl', df_test_nulls_for_iterative_imputer)

Heating_encoder = OneHotEncoder(handle_unknown="ignore", categories=[['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall']])
Heating_encoder.fit(df[['Heating']])

df = One_Hot_Encode_column(Heating_encoder, 'Heating', df)
df_test = One_Hot_Encode_column(Heating_encoder, 'Heating', df_test)
df_nulls_for_iterative_imputer = One_Hot_Encode_column(Heating_encoder, 'Heating', df_nulls_for_iterative_imputer)
df_test_nulls_for_iterative_imputer = One_Hot_Encode_column(Heating_encoder, 'Heating', df_test_nulls_for_iterative_imputer)


Electrical_encoder = OneHotEncoder(handle_unknown="ignore", categories=[['FuseA', 'FuseF', 'FuseP', 'Mix', 'SBrkr']])
Electrical_encoder.fit(df[['Electrical']])

df = One_Hot_Encode_column(Electrical_encoder, 'Electrical', df)
df_test = One_Hot_Encode_column(Electrical_encoder, 'Electrical', df_test)

electrical_nulls = df_nulls_for_iterative_imputer[df_nulls_for_iterative_imputer['Electrical'].isnull()==True]
df_nulls_for_iterative_imputer = One_Hot_Encode_column(Electrical_encoder, 'Electrical', df_nulls_for_iterative_imputer)
df_nulls_for_iterative_imputer.loc[electrical_nulls.index, Electrical_encoder.get_feature_names_out()] = np.nan

df_test_nulls_for_iterative_imputer = One_Hot_Encode_column(Electrical_encoder, 'Electrical', df_test_nulls_for_iterative_imputer)


MSSubClass_encoder = OneHotEncoder(handle_unknown="ignore", categories=[['20','30','40','45','50','60','70','75','80','85','90','120','150','160','180', '190']])
MSSubClass_encoder.fit(df[['MSSubClass']])

df = One_Hot_Encode_column(MSSubClass_encoder, 'MSSubClass', df)
df_test = One_Hot_Encode_column(MSSubClass_encoder, 'MSSubClass', df_test)
df_nulls_for_iterative_imputer = One_Hot_Encode_column(MSSubClass_encoder, 'MSSubClass', df_nulls_for_iterative_imputer)
df_test_nulls_for_iterative_imputer = One_Hot_Encode_column(MSSubClass_encoder, 'MSSubClass', df_test_nulls_for_iterative_imputer)

# MSZoning has nulls in df_test_nulls_for_iterative_imputer - handle manually
MSZoning_encoder = OneHotEncoder(handle_unknown="ignore")
MSZoning_encoder.fit(df[['MSZoning']])

df = One_Hot_Encode_column(MSZoning_encoder, 'MSZoning', df)
df_test = One_Hot_Encode_column(MSZoning_encoder, 'MSZoning', df_test)
df_nulls_for_iterative_imputer = One_Hot_Encode_column(MSZoning_encoder, 'MSZoning', df_nulls_for_iterative_imputer)

MSZoning_nulls = df_test_nulls_for_iterative_imputer[df_test_nulls_for_iterative_imputer['MSZoning'].isnull()==True]
df_test_nulls_for_iterative_imputer = One_Hot_Encode_column(MSZoning_encoder, 'MSZoning', df_test_nulls_for_iterative_imputer)
df_test_nulls_for_iterative_imputer.loc[MSZoning_nulls.index, MSZoning_encoder.get_feature_names_out()] = np.nan


# MasVnrType has nulls in df_test_nulls_for_iterative_imputer - handle manually
MasVnrType_encoder = OneHotEncoder(handle_unknown="ignore")
MasVnrType_encoder.fit(df[['MasVnrType']])

df = One_Hot_Encode_column(MasVnrType_encoder, 'MasVnrType', df)
df_test = One_Hot_Encode_column(MasVnrType_encoder, 'MasVnrType', df_test)
df_nulls_for_iterative_imputer = One_Hot_Encode_column(MasVnrType_encoder, 'MasVnrType', df_nulls_for_iterative_imputer)

MasVnrType_nulls = df_test_nulls_for_iterative_imputer[df_test_nulls_for_iterative_imputer['MasVnrType'].isnull()==True]
df_test_nulls_for_iterative_imputer = One_Hot_Encode_column(MasVnrType_encoder, 'MasVnrType', df_test_nulls_for_iterative_imputer)
df_test_nulls_for_iterative_imputer.loc[MasVnrType_nulls.index, MasVnrType_encoder.get_feature_names_out()] = np.nan


# others will automatically define themselves
to_impute_cols = ['Street', 'LandContour', 'LotConfig', 'Neighborhood', 'BldgType', 'RoofStyle', 
             'Foundation', 'Functional', 'GarageType', 'SaleType', 'SaleCondition']


for column in to_impute_cols:
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(df[[column]])

    df = One_Hot_Encode_column(encoder, column, df)
    df_test = One_Hot_Encode_column(encoder, column, df_test)
    df_nulls_for_iterative_imputer = One_Hot_Encode_column(encoder, column, df_nulls_for_iterative_imputer)
    df_test_nulls_for_iterative_imputer = One_Hot_Encode_column(encoder, column, df_test_nulls_for_iterative_imputer)


# ------------------------------- Imputing missing values ---------------------------------

# Calculate the correlation matrix
correlation_matrix = df.corr()

# we will use KNNImputer and determine the optimal value for n_neighbours using correlation
# matrix
# First column to impute: BsmtExposure

# Get the correlations of the target column with other columns
correlations_with_target = correlation_matrix['BsmtExposure']

# Sort the correlations in descending order (excluding the target column itself)
correlations_with_target = correlations_with_target.sort_values(ascending=False).drop('BsmtExposure')

# we will use the highest correlation values regarding basement 
BsmtExposure_imputer = KNNImputer(n_neighbors=4)
df_for_BsmtExposure_imputer = df[correlations_with_target.index[:4]]
BsmtExposure_imputed = pd.DataFrame(BsmtExposure_imputer.fit_transform(df_nulls_for_iterative_imputer), columns=df_nulls_for_iterative_imputer.columns)

# Fit and transform the DataFrame using the KNNImputer
# only specific columns!!!
#imputed_data = imputer.fit_transform(df_nulls_for_iterative_imputer)


















