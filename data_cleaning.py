import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
#from scipy import stats

# 1 group the data ------------------------------------------------------------

#                   --- house type ---
# MSSubClass: Identifies the type of dwelling involved in the sale.
# MSZoning: The general zoning classification
# BldgType: Type of dwelling
# HouseStyle: Style of dwelling
# OverallQual: Overall material and finish quality
# OverallCond: Overall condition rating

#                   --- property summary ---
# LotFrontage: Linear feet of street connected to property
# LotArea: Lot size in square feet
# LotShape: General shape of property
# LandContour: Flatness of the property
# LotConfig: Lot configuration
# LandSlope: Slope of property

#                   --- neighbourhood ---
# Street: Type of road access
# Alley: Type of alley access
# Neighborhood: Physical locations within Ames city limits
# Condition1: Proximity to main road or railroad
# Condition2: Proximity to main road or railroad (if a second is present)

#                   --- house age ---
# YearBuilt: Original construction date
# YearRemodAdd: Remodel date
# MoSold: Month Sold
# YrSold: Year Sold

#                   --- Exterior design and material quality ---
# RoofStyle: Type of roof
# RoofMatl: Roof material
# Exterior1st: Exterior covering on house
# Exterior2nd: Exterior covering on house (if more than one material)
# MasVnrType: Masonry veneer type
# MasVnrArea: Masonry veneer area in square feet
# ExterQual: Exterior material quality
# ExterCond: Present condition of the material on the exterior
# Foundation: Type of foundation
# Fence: Fence quality
# PavedDrive: Paved driveway


#                   --- Basement ---
# BsmtQual: Height of the basement
# BsmtCond: General condition of the basement
# BsmtExposure: Walkout or garden level basement walls
# BsmtFinType1: Quality of basement finished area
# BsmtFinSF1: Type 1 finished square feet
# BsmtFinType2: Quality of second finished area (if present)
# BsmtFinSF2: Type 2 finished square feet
# BsmtUnfSF: Unfinished square feet of basement area
# BsmtFullBath: Basement full bathrooms
# BsmtHalfBath: Basement half bathrooms
# TotalBsmtSF: Total square feet of basement area

#                   --- Utilities and quality ---
# Utilities: Type of utilities available
# Heating: Type of heating
# HeatingQC: Heating quality and condition
# CentralAir: Central air conditioning
# Electrical: Electrical system
# Fireplaces: Number of fireplaces
# FireplaceQu: Fireplace quality
# Functional: Home functionality rating

#                   --- Floors square and quality ---
# 1stFlrSF: First Floor square feet
# 2ndFlrSF: Second floor square feet
# LowQualFinSF: Low quality finished square feet (all floors)
# GrLivArea: Above grade (ground) living area square feet

#                   --- Rooms count and quality ---
# FullBath: Full bathrooms above grade
# HalfBath: Half baths above grade
# Bedroom: Number of bedrooms above basement level
# Kitchen: Number of kitchens
# KitchenQual: Kitchen quality
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

#                   --- Garage(s) ---
# GarageType: Garage location
# GarageYrBlt: Year garage was built
# GarageFinish: Interior finish of the garage
# GarageCars: Size of garage in car capacity
# GarageArea: Size of garage in square feet
# GarageQual: Garage quality
# GarageCond: Garage condition

#                   --- Pool ---
# PoolArea: Pool area in square feet
# PoolQC: Pool quality

#                   --- Porch ---
# WoodDeckSF: Wood deck area in square feet
# OpenPorchSF: Open porch area in square feet
# EnclosedPorch: Enclosed porch area in square feet
# 3SsnPorch: Three season porch area in square feet
# ScreenPorch: Screen porch area in square feet


#                   --- Other ---
# MiscFeature: Miscellaneous feature not covered in other categories
# MiscVal: $Value of miscellaneous feature

#                   --- Sale ---
# SaleType: Type of sale
# SaleCondition: Condition of sale

#                   --- Target variable ---
# SalePrice - sale price in dollars; target variable

df = pd.read_csv("train.csv", index_col="Id")

# 2 analyze and structure the data --------------------------------------------

# analyze data group by group
description_numerical = df.select_dtypes(include='number').describe().T
description_categorical = df.select_dtypes(include='object').describe().T

def get_categorical_stats(col_name):
    print(df[col_name].unique())
    print(df[col_name].value_counts())
    print()

def get_nulls(df):
    null_columns = df.isnull().sum()
    null_columns = null_columns[null_columns>0]
    df_nulls = pd.DataFrame({"null_count": null_columns, "null_percent": (null_columns/df.shape[0])*100}).sort_values(by='null_count', ascending=False)
    print(df_nulls)
    plt.figure(figsize=(6, 13))
    sns.heatmap(df.isnull().T, yticklabels=df.columns, xticklabels=False, cbar=False, cmap='viridis')
    return df_nulls

df_nulls = get_nulls(df)

#                   --- house type ---
# MSSubClass: should be nominal category; no nulls
# MSZoning: should be nominal category; no nulls
# BldgType: should be nominal; no nulls
# HouseStyle: should be nominal; no nulls
# OverallQual: ordinal; no nulls
# OverallCond: ordinal; no nulls
get_categorical_stats('MSSubClass')
get_categorical_stats('MSZoning')
get_categorical_stats('BldgType')
get_categorical_stats('HouseStyle')
get_categorical_stats('OverallQual')
get_categorical_stats('OverallCond')

sns.set(style="whitegrid")

def barplot_the_y_by_x(x, y, x_label, y_label):
    fig, ax = plt.subplots()
    sns.barplot(x=x.dropna(), y=y, ax=ax)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} by {x_label}")
    
    plt.show()
    
barplot_the_y_by_x(df['MSSubClass'], df['SalePrice'], "MSSubClass", "SalePrice") 
# best are  '60	2-STORY 1946 & NEWER', '75 2-1/2 STORY ALL AGES',  
# '40 1-STORY W/FINISHED ATTIC ALL AGES' and '120 1-STORY PUD (Planned Unit Development) - 1946 & NEWER'
barplot_the_y_by_x(df['MSZoning'], df['SalePrice'], "MSZoning", "SalePrice")     
# best are 'Floating Village Residential' and 'Residential Low Density'
# (living on the water) and (small number of individuals, living in the same building)
barplot_the_y_by_x(df['BldgType'], df['SalePrice'], "BldgType", "SalePrice")   
# best are 'Single-family Detached' and 'Townhouse End Unit'
# (is completely separated by open space on all sides) and 
# (at the end of a row of homes; means you only share a wall on one side)
barplot_the_y_by_x(df['HouseStyle'], df['SalePrice'], "HouseStyle", "SalePrice")
# the best is 'Two and one-half story: 2nd level finished' and 'Two story'
# (the largest amoont of floors all of which are finished) and (Two story)
barplot_the_y_by_x(df['OverallQual'], df['SalePrice'], "OverallQual", "SalePrice")
# the better the quality the bigger the price
barplot_the_y_by_x(df['OverallCond'], df['SalePrice'], "OverallCond", "SalePrice")
# it seems the overall condition is more subjective. the best are '2', '9' and '5'
# we could try to drop it and see if it helps the model performance



#                   --- property summary ---
# LotFrontage: continuous; nulls are present; outliers are present; normal distribution
# LotArea: continuous; no nulls; outliers are present; normal distribution
# LotShape: should be ordinal; no nulls
# LandContour: should be nominal; no nulls
# LotConfig: should be nominal; no nulls
# LandSlope: 
get_categorical_stats('LotFrontage')
plt.hist(df['LotFrontage'], bins=31, edgecolor="black")
plt.xlabel('Lot Frontage')
plt.ylabel('Frequency')
plt.title('Distribution of Lot Frontage')
plt.show()

# understand the nature of nulls, may be for some specific fields it's missing
# or it's completely random
LotFrontage_nulls = df[df['LotFrontage'].isnull()].drop('LotFrontage', axis=1)
for col in LotFrontage_nulls.columns:
    value_counts = LotFrontage_nulls[col].value_counts()
    value_counts.plot(kind='bar')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f'Value Counts of {col} (nulls)')
    plt.show()
# nulls are present for all pave streets, all allpublic utilities, all normal condition2,
# the later it was built the more missing values we have, for all GasA heating, 
# for all without low quality finished square fit, the later the garage was built
# the more missing values we have, for all without three season porch area,
# for all without a pool
# Summary: the later a house was build the more null values it has.
# Our years range is [1872, 2010] and missing values start from 1916. So I presume 
# it just simply hasn't been specified yet or just missing. Therefore, the behavior 
# of null values is not random and I can use grouping by 'year built' column to fill 
#nulls with either mean, median or mode.

# Before we do this, we need to remove outliers for the stat char-s not to be messed up
get_categorical_stats('LotFrontage')
plt.hist(df['LotFrontage'].dropna(), bins=31, edgecolor="black")
plt.xlabel('Lot Frontage')
plt.ylabel('Frequency')
plt.title('Distribution of Lot Frontage')
plt.show()
# ---
plt.boxplot(df['LotFrontage'].dropna())
plt.xlabel('Lot Frontage')
plt.ylabel('Value')
plt.title('Boxplot of Lot Frontage')
plt.show()

df['LotFrontage'] = df.loc[df['LotFrontage']<160]['LotFrontage']

# group by mean year built
grouped_by_year_built = df.groupby('YearBuilt')['LotFrontage'].mean()
plt.plot(grouped_by_year_built)
plt.show()

# fillna with mean
df['LotFrontage'] = df.groupby('YearBuilt')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

# y dependency
# fig, ax = plt.subplots()
# sns.barplot(x=df['LotFrontage'], y=df['SalePrice'], ax=ax, color='black', edgecolor='black', errcolor='red')
# ax.set_xlabel('LotFrontage')
# ax.set_ylabel('SalePrice')
# ax.set_title('SalePrice by LotFrontage')
# ax.set_xticks(int(ax.get_xticks()[::50]))
# plt.show()





get_categorical_stats('LotArea')

fig, ax = plt.subplots()
sns.barplot(x=df['LotArea'].dropna(), y=df['SalePrice'], ax=ax, edgecolor='black', errcolor='red')
ax.set_xlabel('LotArea')
ax.set_ylabel('SalePrice')
ax.set_title('SalePrice by LotArea')
ax.set_xticks(ax.get_xticks()[::100])
plt.show()
# the bigger the area the bigger the price
# ---
plt.hist(df['LotArea'], bins=61, edgecolor="black")
plt.xlabel('Lot LotArea')
plt.ylabel('Frequency')
plt.title('Distribution of Lot Area')
plt.show()
# ---
plt.boxplot(df['LotArea'])
plt.xlabel('Lot Area')
plt.ylabel('Value')
plt.title('Boxplot of Lot Area')
plt.show()

# outliers
df['LotArea'] = df.loc[df['LotArea']<60000]['LotArea']


get_categorical_stats('LotShape')
LotShape_dict = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
df['LotShape'] = [LotShape_dict[l] for l in df['LotShape']]

get_categorical_stats('LandContour')
get_categorical_stats('LotConfig')
get_categorical_stats('LandSlope')


barplot_the_y_by_x(df['LotShape'], df['SalePrice'], "LotShape", "SalePrice")
# best is moderately irregular
barplot_the_y_by_x(df['LandContour'], df['SalePrice'], "LandContour", "SalePrice")
# best is Hillside
barplot_the_y_by_x(df['LotConfig'], df['SalePrice'], "LotConfig", "SalePrice")
# best are Frontage on 3 sides of property and Cul-de-sac
# (3 sides frontage) and (deadend)
barplot_the_y_by_x(df['LandSlope'], df['SalePrice'], "LandSlope", "SalePrice")
# the bigger slope the bigger price


#                   --- neighbourhood ---
# Street: should be nominal; no nulls
# Alley: should be nominal; nulls are present
# Neighborhood: should be nominal; no nulls
# Condition1: condition1 and condition2 have the same values, nominal should be only one
# Condition2: condition1 and condition2 have the same values, nominal should be only one
    
get_categorical_stats('Street')

get_categorical_stats('Alley')
# nulls here stand for 'no alley'
df['Alley'] = df['Alley'].fillna("NA")

get_categorical_stats('Neighborhood').sort_values(ascending=False)
get_categorical_stats('Condition1')
get_categorical_stats('Condition2')


barplot_the_y_by_x(df['Street'], df['SalePrice'], "Street", "SalePrice")
# pave streets are better
barplot_the_y_by_x(df['Alley'], df['SalePrice'], "Alley", "SalePrice")
# paves are better, but may not play a role
barplot_the_y_by_x(df['Neighborhood'], df['SalePrice'], "Neighborhood", "SalePrice")
# best are North Ames, College Creek and Old Town

# condition1 and condition2 have the same values, nominal should be only one
encoder = OneHotEncoder() # Initialize the OneHotEncoder

encoded_cond1 = encoder.fit_transform(df[['Condition1']])
original_cond1_names = encoder.get_feature_names_out(['Condition1'])
encoded_cond1_df = pd.DataFrame(encoded_cond1.toarray(), columns=original_cond1_names)

encoded_cond2 = encoder.fit_transform(df[['Condition2']])
original_cond2_names = encoder.get_feature_names_out(['Condition2'])
encoded_cond2_df = pd.DataFrame(encoded_cond2.toarray(), columns=original_cond2_names)

def df_union(df1, df2, prefix2, on=1):
    combined_df = df1.copy()
    
    for i, row in df1.iterrows():
        for col_name, _ in row.items():
            df2_col_name = prefix2+"_"+col_name.split("_")[1]
            if df2_col_name in df2.columns.values:
                if df2.loc[i][df2_col_name] == 1:
                    combined_df.loc[i][col_name] = 1
        
    return combined_df


encoded_conditions_df = df_union(encoded_cond1_df, encoded_cond2_df, "Condition2")
df = df.drop(["Condition1", 'Condition2'], axis=1)
df = df.append(encoded_conditions_df, ignore_index=True)

barplot_the_y_by_x(df['Condition1'], df['SalePrice'], "Condition1", "SalePrice")
# best is Adjacent to postive off-site feature
barplot_the_y_by_x(df['Condition2'], df['SalePrice'], "Condition2", "SalePrice")
# best are Adjacent to postive off-site feature and Near positive off-site feature--park, greenbelt, etc.


#                   --- house age ---
# YearBuilt: continuous; no nulls, exponential  distribution, outliers are present
# YearRemodAdd: needs to be simplified to nominal yes/no; no nulls
# MoSold: useless
# YrSold: we will use it to compute house age at the moment of selling
    
get_categorical_stats('YearBuilt')
get_categorical_stats('YearRemodAdd')
get_categorical_stats('MoSold')
get_categorical_stats('YrSold')

df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df = df.drop('YrSold', axis=1)
df = df.drop('MoSold', axis=1)
df = df.drop('YearBuilt', axis=1)


fig, ax = plt.subplots()
sns.barplot(x=df['HouseAge'], y=df['SalePrice'], ax=ax, errorbar=None)
ax.set_xticks(ax.get_xticks()[::10])
ax.set_xlabel('HouseAge')
ax.set_ylabel('SalePrice')
ax.set_title("SalePrice by HouseAge")
plt.show()
# the less the age the bigger the price




# YearRemodAdd is the same if there was no remodelling, so we will make 2
# new columns it was (1) or wasn't (0) remodelled and drop this one

df['isRemodelled'] = df.apply(lambda x: 0 if x['YearRemodAdd'] == x['YearBuilt'] else 1, axis=1)
df = df.drop('YearRemodAdd', axis=1)

barplot_the_y_by_x(df['isRemodelled'], df['SalePrice'], "isRemodelled", "SalePrice")
# it may not be useful in the model
barplot_the_y_by_x(df['MoSold'], df['SalePrice'], "MoSold", "SalePrice")
# useless

barplot_the_y_by_x(df['YrSold'], df['SalePrice'], "YrSold", "SalePrice")
# useless


#                   --- Exterior design and material quality ---
# RoofStyle: should be nominal, no nulls
# RoofMatl: should be nominal, no nulls
# Exterior1st AND Exterior2nd: have same values, should be nominal, no nulls
# MasVnrType: should be nominal, nulls are present
# MasVnrArea: continuous, no nulls, exponential distribution, no outliers
# ExterQual: ordinal, no nulls
# ExterCond: ordinal, no nulls
# Foundation: nominal, no nulls
# Fence: nominal, nulls are present
# PavedDrive: nominal, no nulls
    
get_categorical_stats('RoofStyle')
get_categorical_stats('RoofMatl')
get_categorical_stats('Exterior1st')
get_categorical_stats('Exterior2nd')
get_categorical_stats('MasVnrType')
get_categorical_stats('MasVnrArea')
get_categorical_stats('ExterQual')
get_categorical_stats('ExterCond')
get_categorical_stats('Foundation')
get_categorical_stats('Fence')
get_categorical_stats('PavedDrive')

    
barplot_the_y_by_x(df['RoofStyle'], df['SalePrice'], "RoofStyle", "SalePrice")
# best are Shed and hip
barplot_the_y_by_x(df['RoofMatl'], df['SalePrice'], "RoofMatl", "SalePrice")
# best is Wood Shingles
fig, ax = plt.subplots()
sns.barplot(x=df['Exterior1st'], y=df['SalePrice'], ax=ax)
ax.set_xticklabels(df['Exterior1st'].unique(), rotation=45)
ax.set_xlabel('Exterior1st')
ax.set_ylabel('SalePrice')
ax.set_title("SalePrice by Exterior1st")
plt.show()
# best are Imitation Stucco and Stone
fig, ax = plt.subplots()
sns.barplot(x=df['Exterior2nd'], y=df['SalePrice'], ax=ax)
ax.set_xticklabels(df['Exterior2nd'].unique(), rotation=45)
ax.set_xlabel('Exterior2nd')
ax.set_ylabel('SalePrice')
ax.set_title("SalePrice by Exterior2nd")
plt.show()
# best are other and Imitation Stucco
# as they have the same values we will make these values categories (nominal) and
# drop these 2
barplot_the_y_by_x(df['MasVnrType'], df['SalePrice'], "MasVnrType", "SalePrice")
# best is stone
# nulls
df_nulls.loc['MasVnrType']
# 8 nulls. drop them
df = df.dropna(subset=['MasVnrType'])

fig, ax = plt.subplots()
sns.barplot(x=df['MasVnrArea'], y=df['SalePrice'], ax=ax, errorbar=None, edgecolor='black')
ax.set_xticks(ax.get_xticks()[::50])
ax.set_xlabel('MasVnrArea')
ax.set_ylabel('SalePrice')
ax.set_title("SalePrice by MasVnrArea")
plt.show()
# the bigger the area the bigger the price
plt.hist(df['MasVnrArea'], bins=31, edgecolor="black")
plt.xlabel('MasVnrArea')
plt.ylabel('Frequency')
plt.title('Distribution of MasVnrArea')
plt.show()

barplot_the_y_by_x(df['ExterQual'], df['SalePrice'], "ExterQual", "SalePrice")
# best is Excellent; the better is quality the better is price
barplot_the_y_by_x(df['ExterCond'], df['SalePrice'], "ExterCond", "SalePrice")
# best are Average/Typical and good
barplot_the_y_by_x(df['Foundation'], df['SalePrice'], "Foundation", "SalePrice")
# best are Poured Contrete and Wood
barplot_the_y_by_x(df['Fence'], df['SalePrice'], "Fence", "SalePrice")
# best is Good Privacy, may not be useful
# nulls are 'no fence'
df['Fence'] = df['Fence'].fillna("NA")

barplot_the_y_by_x(df['PavedDrive'], df['SalePrice'], "PavedDrive", "SalePrice")
# the best is paved

#                   --- Basement ---
# BsmtQual: should be ordinal without 'NA', nulls are present
# BsmtCond: should be ordinal without 'NA', nulls are present
# BsmtExposure: should be ordinal without 'NA', nulls are present
# BsmtFinType1: should be ordinal without 'NA'
# BsmtFinSF1: 
# BsmtFinType2: should be ordinal without 'NA'
# BsmtFinSF2: 
# BsmtUnfSF: 
# BsmtFullBath: 
# BsmtHalfBath: 
# TotalBsmtSF: 
    
get_categorical_stats('BsmtQual')
get_categorical_stats('BsmtCond')
get_categorical_stats('BsmtExposure')
get_categorical_stats('BsmtFinType1')
get_categorical_stats('BsmtFinSF1')
get_categorical_stats('BsmtFinType2')
get_categorical_stats('BsmtFinSF2')
get_categorical_stats('BsmtUnfSF')
get_categorical_stats('BsmtFullBath')
get_categorical_stats('BsmtHalfBath')
get_categorical_stats('TotalBsmtSF')

# basement quality, condition, exposure, BsmtFinType1, BsmtFinType2 have
# redundant information 'no basement'
# create new column and make those columns ordinal
     
barplot_the_y_by_x(df['BsmtQual'], df['SalePrice'], "BsmtQual", "SalePrice")
# best is excellent
# nulls here are 'no basement'
df['BsmtQual'] = df['BsmtQual'].fillna(0)
barplot_the_y_by_x(df['BsmtCond'], df['SalePrice'], "BsmtCond", "SalePrice")
# best is Good
# nulls here are 'no basement'
df['BsmtCond'] = df['BsmtCond'].fillna(0)
barplot_the_y_by_x(df['BsmtExposure'], df['SalePrice'], "BsmtExposure", "SalePrice")
# nulls here are 'no basement'
df['BsmtExposure'] = df['BsmtExposure'].fillna(0)

df['isBasement'] = df['BsmtExposure'].apply(lambda x: 0 if x=="NA" else 1)
#df['BsmtQual'] = 

barplot_the_y_by_x(df['BsmtFinType1'], df['SalePrice'], "BsmtFinType1", "SalePrice")
#
barplot_the_y_by_x(df['BsmtFinSF1'], df['SalePrice'], "BsmtFinSF1", "SalePrice")
#
barplot_the_y_by_x(df['BsmtFinType2'], df['SalePrice'], "BsmtFinType2", "SalePrice")
#
barplot_the_y_by_x(df['BsmtFinSF2'], df['SalePrice'], "BsmtFinSF2", "SalePrice")
#
barplot_the_y_by_x(df['BsmtUnfSF'], df['SalePrice'], "BsmtUnfSF", "SalePrice")
#
barplot_the_y_by_x(df['BsmtFullBath'], df['SalePrice'], "BsmtFullBath", "SalePrice")
#
barplot_the_y_by_x(df['BsmtHalfBath'], df['SalePrice'], "BsmtHalfBath", "SalePrice")
#
barplot_the_y_by_x(df['TotalBsmtSF'], df['SalePrice'], "TotalBsmtSF", "SalePrice")
#


#                   --- Utilities and quality ---
# Utilities: Type of utilities available
# Heating: Type of heating
# HeatingQC: Heating quality and condition
# CentralAir: Central air conditioning
# Electrical: Electrical system
# Fireplaces: Number of fireplaces
# FireplaceQu: Fireplace quality
# Functional: Home functionality rating

#                   --- Floors square and quality ---
# 1stFlrSF: First Floor square feet
# 2ndFlrSF: Second floor square feet
# LowQualFinSF: Low quality finished square feet (all floors)
# GrLivArea: Above grade (ground) living area square feet

#                   --- Rooms count and quality ---
# FullBath: Full bathrooms above grade
# HalfBath: Half baths above grade
# Bedroom: Number of bedrooms above basement level
# Kitchen: Number of kitchens
# KitchenQual: Kitchen quality
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

#                   --- Garage(s) ---
# GarageType: Garage location
# GarageYrBlt: Year garage was built
# GarageFinish: Interior finish of the garage
# GarageCars: Size of garage in car capacity
# GarageArea: Size of garage in square feet
# GarageQual: Garage quality
# GarageCond: Garage condition

#                   --- Pool ---
# PoolArea: Pool area in square feet
# PoolQC: Pool quality

#                   --- Porch ---
# WoodDeckSF: Wood deck area in square feet
# OpenPorchSF: Open porch area in square feet
# EnclosedPorch: Enclosed porch area in square feet
# 3SsnPorch: Three season porch area in square feet
# ScreenPorch: Screen porch area in square feet


#                   --- Other ---
# MiscFeature: Miscellaneous feature not covered in other categories
# MiscVal: $Value of miscellaneous feature

#                   --- Sale ---
# SaleType: Type of sale
# SaleCondition: Condition of sale

#                   --- Target variable ---
# SalePrice - sale price in dollars; target variable






def get_categorical_stats(col_name):
    print(df[col_name].unique())
    print(df[col_name].value_counts())
    print()
  

 
for col in df_nulls.index:
    get_categorical_stats(col)

# the data has a lot of null values, but speaking of those that are categorical
# according to the docs, it means 'no pool', 'no alley access', 'no fence' etc.
# so we will handle those first

# PoolQC: Pool qualityEx
# Ex Excellent,
# Gd Good,
# TA Average/Typical,
# Fa Fair,
# NA No Pool
df['PoolQC'] = df['PoolQC'].fillna("NA")

# MiscFeature: Miscellaneous feature not covered in other categories
# Elev	Elevator - no elevator values in the train dataset
# Gar2	2nd Garage (if not described in garage section) - we should describe it there and remove this value
# Othr	Other - only 2 rows with it, may be delete it
# Shed	Shed (over 100 SF) - make a different column for shed 1/0
# TenC	Tennis Court - only one row with it - delete it
# NA	None
df['MiscFeature'] = df['MiscFeature'].fillna("NA")
double_garage = df[df['MiscFeature']=="Gar2"].T
# 1
# 'GarageType' does have a value '2Types' for multiple garages, we should make
# 'GarageType' nominal
df['GarageType'] = df['GarageType'].fillna("NA")
encoder = OneHotEncoder(sparse_output=False) # Initialize the OneHotEncoder
encoded_data = encoder.fit_transform(df[['GarageType']]) # Fit and transform the data
column_names = encoder.get_feature_names_out(['GarageType']) # Create column names for the encoded variables
df[column_names] = encoded_data # Concatenate the encoded columns to the original DataFrame

# save these 2 values of 'double_garage' there as well
# Assign values to column 'GarageType_2Types' based on values in column 'MiscFeature'
df.loc[df['MiscFeature']=="Gar2", "GarageType_2Types"] = 1


# 2
# make a different column for shed 1/0
df['Shed'] = df['MiscFeature'].apply(lambda x: 1 if x == 'Shed' else 0)

# 3
# drop the 'MiscFeature' column as it doesn't help anymore
df = df.drop('MiscFeature', axis=1)

# Alley: Type of alley access to property
# Grvl	Gravel
# Pave	Paved
# NA 	No alley access
df['Alley'] = df['Alley'].fillna("NA")

# Fence: Fence quality
# 		
# GdPrv	Good Privacy
# MnPrv	Minimum Privacy
# GdWo	Good Wood
# MnWw	Minimum Wood/Wire
# NA	No Fence
df['Fence'] = df['Fence'].fillna("NA")

# FireplaceQu: Fireplace quality

# Ex	Excellent - Exceptional Masonry Fireplace
# Gd	Good - Masonry Fireplace in main level
# TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
# Fa	Fair - Prefabricated Fireplace in basement
# Po	Poor - Ben Franklin Stove
# NA	No Fireplace
df['FireplaceQu'] = df['FireplaceQu'].fillna("NA")

# LotFrontage: Linear feet of street connected to property
# this field is numerical, so handling null values requires additional analysis
#sns.histplot(df['LotFrontage'].dropna(), kde=True)

# GarageYrBlt: Year garage was built
# this field is numerical, so handling null values requires additional analysis

# GarageFinish: Interior finish of the garage

# Fin	Finished
# RFn	Rough Finished	
# Unf	Unfinished
# NA	No Garage
df['GarageFinish'] = df['GarageFinish'].fillna("NA")


# GarageQual: Garage quality

# Ex	Excellent
# Gd	Good
# TA	Typical/Average
# Fa	Fair
# Po	Poor
# NA	No Garage
df['GarageQual'] = df['GarageQual'].fillna("NA")


# GarageCond: Garage condition

# Ex	Excellent
# Gd	Good
# TA	Typical/Average
# Fa	Fair
# Po	Poor
# NA	No Garage
df['GarageCond'] = df['GarageCond'].fillna("NA")


# BsmtExposure: Refers to walkout or garden level walls

# Gd	Good Exposure
# Av	Average Exposure (split levels or foyers typically score average or above)	
# Mn	Mimimum Exposure
# No	No Exposure
# NA	No Basement
df['BsmtExposure'] = df['BsmtExposure'].fillna("NA")


# BsmtFinType2: Rating of basement finished area (if multiple types)

# GLQ	Good Living Quarters
# ALQ	Average Living Quarters
# BLQ	Below Average Living Quarters	
# Rec	Average Rec Room
# LwQ	Low Quality
# Unf	Unfinshed
# NA	No Basement
df['BsmtFinType2'] = df['BsmtFinType2'].fillna("NA")


# BsmtQual: Evaluates the height of the basement

# Ex	Excellent (100+ inches)	
# Gd	Good (90-99 inches)
# TA	Typical (80-89 inches)
# Fa	Fair (70-79 inches)
# Po	Poor (<70 inches
# NA	No Basement
df['BsmtQual'] = df['BsmtQual'].fillna("NA")


# BsmtCond: Evaluates the general condition of the basement

# Ex	Excellent
# Gd	Good
# TA	Typical - slight dampness allowed
# Fa	Fair - dampness or some cracking or settling
# Po	Poor - Severe cracking, settling, or wetness
# NA	No Basement
df['BsmtCond'] = df['BsmtCond'].fillna("NA")


# BsmtFinType1: Rating of basement finished area

# GLQ	Good Living Quarters
# ALQ	Average Living Quarters
# BLQ	Below Average Living Quarters	
# Rec	Average Rec Room
# LwQ	Low Quality
# Unf	Unfinshed
# NA	No Basement
df['BsmtFinType1'] = df['BsmtFinType1'].fillna("NA")


# MasVnrType: Masonry veneer type

# BrkCmn	Brick Common
# BrkFace	Brick Face
# CBlock	Cinder Block
# None	None
# Stone	Stone
df['MasVnrType'] = df['MasVnrType'].fillna("NA")


# MasVnrArea: Masonry veneer area in square feet
# this field is numerical, so handling null values requires additional analysis


# Electrical: Electrical system

# SBrkr	Standard Circuit Breakers & Romex
# FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
# FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
# FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
# Mix	Mixed

# as it contains only one null value, we will delete it
df = df.dropna(subset=['Electrical'])


df_nulls = get_nulls(df)


# deal with outliers
for col in description_numerical.index:
    plt.figure()  # Create a new figure for each box plot
    plt.boxplot(df[col].dropna())
    plt.title(col)  # Set the title of the plot as the column name
    plt.show()  # Display the box plot

# MSSubClass - categorical
# LotFrontage - outliers detected
LotFrontage_count_values = df['LotFrontage'].value_counts()
df = df.loc[df['LotFrontage']<300]

# LotArea - outliers detected
LotArea_count_values = df['LotArea'].value_counts()
df = df.loc[df['LotArea']<71000]

# OverallQual - categorical
# OverallCond - categorical
# YearBuilt - no outliers
# YearRemodAdd - no outliers
# MasVnrArea - outliers detected, also there are a lot of 0es, therefore
# Masonry veneer area is absent there, we will add it as a separate column
MasVnrArea_count_values = df['MasVnrArea'].value_counts()
df = df.loc[df['MasVnrArea']<1300]
df['IsMasVnr'] = df['MasVnrArea'].apply(lambda x: 0 if x == 0 else 1)

#BsmtFinSF1 - outliers detected
BsmtFinSF1_count_values = df['BsmtFinSF1'].value_counts()
# what does 0 mean in this column?
zeroes = df[df['BsmtFinSF1']==0]
#it means it is either unfinished or absent
# the column doesn't have many repetitive values (many 1s)
# and the 'BsmtFinType1' tells us about basement, I'd rather drop the column
df = df.drop('BsmtFinSF1', axis=1)
description_numerical = description_numerical.drop('BsmtFinSF1', axis=0)

#BsmtFinSF2 - outliers detected
BsmtFinSF2_count_values = df['BsmtFinSF2'].value_counts()
# same as with BsmtFinSF1
df = df.drop('BsmtFinSF2', axis=1)
description_numerical = description_numerical.drop('BsmtFinSF2', axis=0)

# BsmtUnfSF
BsmtUnfSF_count_values = df['BsmtUnfSF'].value_counts()
# same as with BsmtFinSF2
df = df.drop('BsmtUnfSF', axis=1)
description_numerical = description_numerical.drop('BsmtUnfSF', axis=0)

for col in description_numerical.index:
    plt.figure()  # Create a new figure for each box plot
    plt.boxplot(df[col].dropna())
    plt.title(col)  # Set the title of the plot as the column name
    plt.show()  # Display the box plot

# TotalBsmtSF - no outliers
TotalBsmtSF_count_values = df['TotalBsmtSF'].value_counts()

# 1stFlrSF - outliers detected
firstFlrSF_count_values = df['1stFlrSF'].value_counts()
df = df.loc[df['1stFlrSF']<3000]

# 2ndFlrSF - outliers detected
secondFlrSF_count_values = df['2ndFlrSF'].value_counts()
df = df.loc[df['2ndFlrSF']<3000]
    
# categorical analysis
# numerical nans research




















