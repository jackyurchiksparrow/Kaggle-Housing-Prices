import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

# na_filter=False because it understood 'NA' as nan
df = pd.read_csv("Post_nulls_datasets/train.csv", na_filter=False)
df_test = pd.read_csv("Post_nulls_datasets/test.csv", na_filter=False)
df_nulls_for_iterative_imputer = pd.read_csv("Post_nulls_datasets/train_iterative_imputer_nulls.csv", na_filter=False)
df_test_nulls_for_iterative_imputer = pd.read_csv("Post_nulls_datasets/test_iterative_imputer_nulls.csv", na_filter=False)

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

def One_Hot_Encode_the_columns(encoder, column, data_frame):
    transformed = encoder.fit_transform(data_frame[column])
    feature_names = encoder.get_feature_names_out(column)
    transformed_df = pd.DataFrame(transformed.toarray(), columns=feature_names)
    data_frame = data_frame.join(transformed_df)
    data_frame.drop(column, inplace=True, axis=1)
    return data_frame


mlb = MultiLabelBinarizer()
nominals = ['MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType',
           'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'Electrical',
           'Functional', 'GarageType', 'SaleType', 'SaleCondition', 'MSSubClass']
nominal_multilabeled = ['Utilities', 'Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd', 
                        'BsmtFinType1', 'BsmtFinType2']
# MSZoning - nominal data
# Street - nominal data
# LotShape - ordinal data; signifies irregularity of shape (bigger number more irregular shape)
values = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
df['LotShape'] = df['LotShape'].apply(lambda x: values[x])
df_test['LotShape'] = df_test['LotShape'].apply(lambda x: values[x])
df_nulls_for_iterative_imputer['LotShape'] = df_nulls_for_iterative_imputer['LotShape'].apply(lambda x: values[x])
df_test_nulls_for_iterative_imputer['LotShape'] = df_test_nulls_for_iterative_imputer['LotShape'].apply(lambda x: values[x])
# LandContour - nominal data
# Utilities - nominal, but needs to be multilabeled
# LotConfig - nominal data
# LandSlope - ordinal (bigger number - biger slope)
values = {'Gtl': 1, 'Mod': 2, 'Sev': 3}
df['LandSlope'] = df['LandSlope'].apply(lambda x: values[x])
# Neighborhood - nominal data
# Condition1, Condition2 - nominal data, but needs to be multilabeled
# BldgType - nominal data
# HouseStyle - nominal data
set(df['HouseStyle'].unique()).symmetric_difference(set(df_test['HouseStyle'].unique()))
# RoofStyle - nominal data
# RoofMatl - nominal data
set(df['RoofMatl'].unique()).symmetric_difference(set(df_test['RoofMatl'].unique()))
# Exterior1st, Exterior2nd - nominal data, but needs to be multilabeled
# MasVnrType - nominal data
# ExterQual - ordinal (bigger number - better quality)
values = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['ExterQual'] = df['ExterQual'].apply(lambda x: values[x])
df_test['ExterQual'] = df_test['ExterQual'].apply(lambda x: values[x])
# ExterCond - ordinal (bigger number - better quality)
df['ExterCond'] = df['ExterCond'].apply(lambda x: values[x])
df_test['ExterCond'] = df_test['ExterCond'].apply(lambda x: values[x])
# HeatingQC - ordinal (bigger number - better quality)
df['HeatingQC'] = df['HeatingQC'].apply(lambda x: values[x])
df_test['HeatingQC'] = df_test['HeatingQC'].apply(lambda x: values[x])
# KitchenQual - ordinal (bigger number - better quality)
df['KitchenQual'] = df['KitchenQual'].apply(lambda x: values[x])
df_test['KitchenQual'] = df_test['KitchenQual'].apply(lambda x: values[x])
# Foundation - nominal data
# BsmtQual - ordinal (bigger number - better quality)
values = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['BsmtQual'] = df['BsmtQual'].apply(lambda x: values[x])
df_test['BsmtQual'] = df_test['BsmtQual'].apply(lambda x: values[x])
# BsmtCond - ordinal (bigger number - better condition)
df['BsmtCond'] = df['BsmtCond'].apply(lambda x: values[x])
df_test['BsmtCond'] = df_test['BsmtCond'].apply(lambda x: values[x])
# FireplaceQu - ordinal (bigger number - better quality)
df['FireplaceQu'] = df['FireplaceQu'].apply(lambda x: values[x])
df_test['FireplaceQu'] = df_test['FireplaceQu'].apply(lambda x: values[x])
# GarageQual - ordinal (bigger number - better quality)
df['GarageQual'] = df['GarageQual'].apply(lambda x: values[x])
df_test['GarageQual'] = df_test['GarageQual'].apply(lambda x: values[x])
# GarageCond - ordinal (bigger number - better quality)
df['GarageCond'] = df['GarageCond'].apply(lambda x: values[x])
df_test['GarageCond'] = df_test['GarageCond'].apply(lambda x: values[x])
# BsmtExposure - ordinal (bigger number - better exposure)
values = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
df['BsmtExposure'] = df['BsmtExposure'].apply(lambda x: values[x])
df_test['BsmtExposure'] = df_test['BsmtExposure'].apply(lambda x: values[x])
# BsmtFinType1, BsmtFinType2 - nominal data, but needs to be multilabeled
# Heating - nominal data
set(df['Heating'].unique()).symmetric_difference(set(df_test['Heating'].unique()))
# CentralAir - ordinal binary
values = {'Y': 1, 'N': 0}
df['CentralAir'] = df['CentralAir'].apply(lambda x: values[x])
df_test['CentralAir'] = df_test['CentralAir'].apply(lambda x: values[x])
# Electrical - nominal data
set(df['Electrical'].unique()).symmetric_difference(set(df_test['Electrical'].unique()))
# Functional - nominal data
# GarageType - nominal data
# GarageFinish - ordinal (bigger number - closer to finish)
values = {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
df['GarageFinish'] = df['GarageFinish'].apply(lambda x: values[x])
df_test['GarageFinish'] = df_test['GarageFinish'].apply(lambda x: values[x])
# PavedDrive - ordinal (bigger number - better pavement)
values = {'Y': 2, 'P': 1, 'N': 0}
df['PavedDrive'] = df['PavedDrive'].apply(lambda x: values[x])
df_test['PavedDrive'] = df_test['PavedDrive'].apply(lambda x: values[x])
# SaleType - nominal data
# SaleCondition - nominal data
# MSSubClass - nominal data
set(df['MSSubClass'].unique()).symmetric_difference(set(df_test['MSSubClass'].unique()))


encoder = OneHotEncoder(handle_unknown='ignore')
df = One_Hot_Encode_the_columns(nominals, df)
df_test = One_Hot_Encode_the_columns(nominals, df_test)
df_nulls_for_iterative_imputer = One_Hot_Encode_the_columns(nominals, df_nulls_for_iterative_imputer)
df_test_nulls_for_iterative_imputer = One_Hot_Encode_the_columns(nominals, df_test_nulls_for_iterative_imputer)



















