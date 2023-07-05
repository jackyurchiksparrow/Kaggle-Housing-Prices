import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv("train.csv", index_col="Id")

df.describe()

# analyze data group by group
description_numerical = df.select_dtypes(include='number').describe().T
description_categorical = df.select_dtypes(include='object').describe().T

def get_categorical_stats(col_name):
    print(df[col_name].unique())
    print(df[col_name].value_counts())
    print()

def barplot_the_salePrice_by_x(x, x_label, x_ticks_step=0, rotation=0, figsize=(8,6)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=x.dropna(), y=df['SalePrice'].dropna(), ax=ax, errorbar=('ci', False))
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('SalePrice')
    ax.set_title(f"SalePrice by {x_label}")
    
    formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}K')
    ax.yaxis.set_major_formatter(formatter)
    
    if x_ticks_step > 0:
        ax.set_xticks(ax.get_xticks()[::x_ticks_step])
        
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation) 
    
    plt.show()
    
    
#                   --- house type ---
# MSSubClass: Identifies the type of dwelling involved in the sale.
# MSZoning: The general zoning classification
# BldgType: Type of dwelling
# HouseStyle: Style of dwelling
# OverallQual: Overall material and finish quality
# OverallCond: Overall condition rating
barplot_the_salePrice_by_x(df['MSSubClass'], "MSSubClass") 
# best are  '60	2-STORY 1946 & NEWER', '75 2-1/2 STORY ALL AGES',  
# '40 1-STORY W/FINISHED ATTIC ALL AGES' and '120 1-STORY PUD (Planned Unit Development) - 1946 & NEWER'
barplot_the_salePrice_by_x(df['MSZoning'], "MSZoning")     
# best are 'Floating Village Residential' and 'Residential Low Density'
# (living on the water) and (small number of individuals, living in the same building)
barplot_the_salePrice_by_x(df['BldgType'], "BldgType")   
# best are 'Single-family Detached' and 'Townhouse End Unit'
# (is completely separated by open space on all sides) and 
# (at the end of a row of homes; means you only share a wall on one side)
barplot_the_salePrice_by_x(df['HouseStyle'], "HouseStyle")
# the best is 'Two and one-half story: 2nd level finished' and 'Two story'
# (the largest amoont of floors all of which are finished) and (Two story)
barplot_the_salePrice_by_x(df['OverallQual'], "OverallQual")
# the better the quality the bigger the price
barplot_the_salePrice_by_x(df['OverallCond'], "OverallCond")
# it seems the overall condition is more subjective. the best are '2', '9' and '5'
# we could try to drop it and see if it helps the model performance


#                   --- property summary ---
# LotFrontage: Linear feet of street connected to property
# LotArea: Lot size in square feet
# LotShape: General shape of property
# LandContour: Flatness of the property
# LotConfig: Lot configuration
# LandSlope: Slope of property
barplot_the_salePrice_by_x(df['LotFrontage'], "LotFrontage", x_ticks_step = 10)
# prices are tend to be higher for larger lot frontage
barplot_the_salePrice_by_x(df['LotArea'], "LotArea", x_ticks_step=50, rotation=45)
# prices are tend to be higher for larger lot area
barplot_the_salePrice_by_x(df['LotShape'], "LotShape")
# prices are tend to be higher around irregular shapes
barplot_the_salePrice_by_x(df['LandContour'], "LandContour")
# best option is Hillside
barplot_the_salePrice_by_x(df['LotConfig'], "LotConfig")
# best are Frontage on 3 sides of property and Cul-de-sac
# (3 sides frontage) and (deadend)
barplot_the_salePrice_by_x(df['LandSlope'], "LandSlope")
# the bigger slope the bigger price

#                   --- neighbourhood ---
# Street: Type of road access
# Alley: Type of alley access
# Neighborhood: Physical locations within Ames city limits
# Condition1: Proximity to main road or railroad
# Condition2: Proximity to main road or railroad (if a second is present)
barplot_the_salePrice_by_x(df['Street'], "Street")
# pave streets are more expensive
barplot_the_salePrice_by_x(df['Alley'], "Alley")
# paves are better
barplot_the_salePrice_by_x(df['Neighborhood'], "Neighborhood", rotation=45)
# best are Northridge, Northridge Heights, Stone Brook
barplot_the_salePrice_by_x(df['Condition1'], "Condition1")
# best are Adjacent to postive off-site feature and Near positive off-site feature--park, 
# greenbelt, etc.
barplot_the_salePrice_by_x(df['Condition2'], "Condition2")
# part of condition 1, should be combined together in categorical_variables_py

#                   --- house age ---
# YearBuilt: Original construction date
# YearRemodAdd: Remodel date
# MoSold: Month Sold
# YrSold: Year Sold
barplot_the_salePrice_by_x(df['YearBuilt'], "YearBuilt", x_ticks_step=5, rotation=45)
# modern building are more expensive (with bigger values of year built)
barplot_the_salePrice_by_x(df['YearRemodAdd'], "YearRemodAdd", x_ticks_step=5, rotation=45)
# the later a house was remodelled the bigger the price
barplot_the_salePrice_by_x(df['MoSold'], "MoSold")
# don't think it has any impact
barplot_the_salePrice_by_x(df['YrSold'], "YrSold")
# don't think it has any impact

df['YrSold'].isnull().any()
df['YearBuilt'].isnull().any()
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
barplot_the_salePrice_by_x(df['HouseAge'], "HouseAge", x_ticks_step=5, rotation=45)
# the younger the house the bigger the price
df.drop(['MoSold', 'YrSold'], axis=1, inplace=True)

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
barplot_the_salePrice_by_x(df['RoofStyle'], "RoofStyle")
# best are Shed and hip
barplot_the_salePrice_by_x(df['RoofMatl'], "RoofMatl")
# best is Wood Shingles
barplot_the_salePrice_by_x(df['Exterior1st'], "Exterior1st", rotation=30)
# best are Imitation Stucco and Stone
barplot_the_salePrice_by_x(df['Exterior2nd'], "Exterior2nd", rotation=30)
# part of Exterior1st, should be combined together in categorical_variables_py
barplot_the_salePrice_by_x(df['MasVnrType'], "MasVnrType")
# best is stone
barplot_the_salePrice_by_x(df['MasVnrArea'].dropna().astype(int), "MasVnrArea", x_ticks_step=20, rotation=15)
# the larger the Masonry area the bigger the price
barplot_the_salePrice_by_x(df['ExterQual'], "ExterQual")
# best is Excellent; the higher is quality the higher is price
barplot_the_salePrice_by_x(df['ExterCond'], "ExterCond")
# best are Average/Typical and good
barplot_the_salePrice_by_x(df['Foundation'], "Foundation")
# best are Poured Contrete and Wood
barplot_the_salePrice_by_x(df['Fence'], "Fence")
# best is Good Privacy
barplot_the_salePrice_by_x(df['PavedDrive'], "PavedDrive")
# the best is paved

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
barplot_the_salePrice_by_x(df['BsmtQual'], "BsmtQual")
# best is excellent
barplot_the_salePrice_by_x(df['BsmtCond'], "BsmtCond")
# best is Good
barplot_the_salePrice_by_x(df['BsmtExposure'], "BsmtExposure")
# best is Good
barplot_the_salePrice_by_x(df['BsmtFinType1'], "BsmtFinType1")
# best is Good Living Quarters
barplot_the_salePrice_by_x(df['BsmtFinType2'], "BsmtFinType2")
# part of BsmtFinType1, should be combined together in categorical_variables_py
barplot_the_salePrice_by_x(df['BsmtFinSF1'], "BsmtFinSF1", x_ticks_step=30, rotation=15, figsize=(10,6))
# the larger is square feet the bigger the price
barplot_the_salePrice_by_x(df['BsmtFinSF2'], "BsmtFinSF2", x_ticks_step=10, figsize=(10,6))
# it's better to combine with BsmtFinSF1 to know all basement sqaure feet
df['BsmtFinSF1'].isnull().any()
df['BsmtFinSF2'].isnull().any()
df['BsmtFinSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
df.drop(['BsmtFinSF1', 'BsmtFinSF2'], axis=1, inplace=True)
barplot_the_salePrice_by_x(df['BsmtFinSF'], "BsmtFinSF", x_ticks_step=30, rotation=15, figsize=(10,6))
# the larger is square feet the bigger the price
barplot_the_salePrice_by_x(df['BsmtUnfSF'], "BsmtUnfSF", x_ticks_step=30, rotation=15, figsize=(10,6))
# larger square feet, (even unfinished) costs more
barplot_the_salePrice_by_x(df['BsmtFullBath'], "BsmtFullBath")
# the optimal choice is 2
barplot_the_salePrice_by_x(df['BsmtHalfBath'], "BsmtHalfBath")
# the more half barthrooms the cheaper (people do not prefer halves bathrooms)
barplot_the_salePrice_by_x(df['TotalBsmtSF'], "TotalBsmtSF", x_ticks_step=30, rotation=15, figsize=(10,6))
# the larger is square feet the bigger the price

#                   --- Utilities and quality ---
# Utilities: Type of utilities available
# Heating: Type of heating
# HeatingQC: Heating quality and condition
# CentralAir: Central air conditioning
# Electrical: Electrical system
# Fireplaces: Number of fireplaces
# FireplaceQu: Fireplace quality
# Functional: Home functionality rating
barplot_the_salePrice_by_x(df['Utilities'], "Utilities")
# more utilities - higher price
barplot_the_salePrice_by_x(df['Heating'], "Heating")
# best are Gas forced warm air furnace and Gas hot water or steam heat
barplot_the_salePrice_by_x(df['HeatingQC'], "HeatingQC")
# better quality - higher price
barplot_the_salePrice_by_x(df['CentralAir'], "CentralAir")
# more expensive with air conditioning that without
barplot_the_salePrice_by_x(df['Electrical'], "Electrical")
# the best is Standard Circuit Breakers & Romex
barplot_the_salePrice_by_x(df['Fireplaces'], "Fireplaces")
# more fireplaces - higher price
barplot_the_salePrice_by_x(df['FireplaceQu'], "FireplaceQu")
# better quality - higher price
barplot_the_salePrice_by_x(df['Functional'], "Functional")
# the best are typical and moderate deductions


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













































