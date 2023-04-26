import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
    
    # IMPORTANT - GIVES US INFORMATION ON THE DATA
    # demonstrateHelpers(trainDF)
    
    # Correlation between the attributes and SalePrice 
    # findCorr(trainDF)
    
    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    #Finds the most accurate k value
    # paramSearchPlot(trainInput, trainOutput)
    # tuneGradientBoostingRegressor(trainInput, trainOutput, predictors)
    
    #The Models
    # doExperiment(trainInput, trainOutput, predictors)
    # doBayesianRidge(trainInput, trainOutput, predictors)
    # doLasso(trainInput, trainOutput, predictors)
    # doElasticNet(trainInput, trainOutput, predictors)
    # doLinearRegression(trainInput, trainOutput, predictors)
    # doGradientBoostingRegressor(trainInput, trainOutput, predictors)
    # doKNeighborsRegressor(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    # visualization(trainInput, predictors)

# ===============================================================================
# ===============================================================================
#                                THE MODELS
# ===============================================================================
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
'''
# def doExperiment(trainInput, trainOutput, predictors):
#     # alg = LinearRegression()
#     #The other models that we tried
#     # alg = KNeighborsRegressor(n_neighbors=7)
#     # alg = ElasticNet()
#     # alg = Lasso()
#     # alg = BayesianRidge()
#     alg = GradientBoostingRegressor(n_estimators=100,learning_rate = 0.1) 
#     cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1, error_score='raise').mean()
#     print("CV Average Score:", cvMeanScore)

def doLinearRegression(trainInput, trainOutput, predictors):
    alg1 = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg1, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1, error_score='raise').mean()
    print("Linear Regression")
    print("CV Average Score:", cvMeanScore)
    # alg2 = LinearRegression(normalize = True)
    # cvMeanScore = model_selection.cross_val_score(alg2, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1, error_score='raise').mean()
    # print("Linear Regression Normalized")
    # print("CV Average Score:", cvMeanScore)
    
'''
Does k-fold CV on the Kaggle training set using KNeighborsRegressor.
'''
def doKNeighborsRegressor(trainInput, trainOutput, predictors):
    alg = KNeighborsRegressor(n_neighbors=7)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1, error_score='raise').mean()
    print("K Nearest Neighbors Regressor")
    print("CV Average Score:", cvMeanScore)

'''
Does k-fold CV on the Kaggle training set using ElasticNet.
'''
def doElasticNet(trainInput, trainOutput, predictors):
    alg = ElasticNet()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1, error_score='raise').mean()
    print("Elastic Net")
    print("CV Average Score:", cvMeanScore)

'''
Does k-fold CV on the Kaggle training set using Lasso.
'''    
def doLasso(trainInput, trainOutput, predictors):
    alg = Lasso()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1, error_score='raise').mean()
    print("Lasso")
    print("CV Average Score:", cvMeanScore)

'''
Does k-fold CV on the Kaggle training set using BayedianRidge.
'''
def doBayesianRidge(trainInput, trainOutput, predictors):
    alg = BayesianRidge()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1, error_score='raise').mean()
    print("Bayesian Ridge")
    print("CV Average Score:", cvMeanScore)

'''
Does k-fold CV on the Kaggle training set using GradientBoostingRegressor.
'''    
def doGradientBoostingRegressor(trainInput, trainOutput, predictors):
    alg = GradientBoostingRegressor(learning_rate=0.04, max_depth=4, n_estimators=170, subsample=0.89)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1, error_score='raise').mean()
    print("Gradient Boosting Regressor")
    print("CV Average Score:", cvMeanScore)


'''
Does k-fold CV on the Kaggle training set using GradientBoostingRegressor.
'''   
def tuneGradientBoostingRegressor(trainInput, trainOutput, predictors):
    # BEGIN: from https://www.youtube.com/watch?v=7VibNwcnr4s
    model = GradientBoostingRegressor()
    parameters = {'learning_rate' : sp_randFloat(),
                  'subsample' : sp_randFloat(),
                  'n_estimators' : sp_randInt(100, 1000),
                  'max_depth' : sp_randInt(4, 10)
                 }
    # EXPLANATION:  used to find the optimal combination of hyper parameters
    #               completes 10 iterations and returns the parameters that gave 
    #               the best score
    randm = RandomizedSearchCV(estimator=model, param_distributions=parameters, 
                               cv=5, n_iter=10, n_jobs=-1)
    randm.fit(trainInput, trainOutput)
    
    #RESULTS FROM RANDOM SEARCH:
    print("\n=====================")
    print("The best estimator:")
    print(randm.best_estimator_)
    print("The best score:")
    print(randm.best_score_)
    print("The best params:")
    print(randm.best_params_)
    # END: from https://www.youtube.com/watch?v=7VibNwcnr4s
    
# ===============================================================================
# ===============================================================================
#                           CSV FILE FOR KAGGLE TEST
# ===============================================================================
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = GradientBoostingRegressor(learning_rate=0.04, max_depth=4, n_estimators=170, subsample=0.89)

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ===============================================================================
# ===============================================================================
#                           PRE-PROCESSING OF THE DATA
# ===============================================================================
# ===============================================================================
'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    
    # ===============================================================================
    #                           ATTRIBUTES FROM EXPERIMENT A
    # ===============================================================================
   
    #Only the numerical attributes that didn't require pre-processing
    # numPredNoPrep = ['MSSubClass', 'LotArea', 'OverallQual'
    #         ,'OverallCond', 'YearBuilt', 'YearRemodAdd'
    #         , '1stFlrSF', '2ndFlrSF'
    #         , 'GrLivArea' 
    #         , 'FullBath'
    #         ,'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd'
    #         ,'Fireplaces' 
    #         , 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea'
    #         ,'MiscVal', 'MoSold', 'YrSold']
    # predictors = numPredNoPrep
    
    #All numerical attributes (without missing values)
    # numPredNoNull = ['MSSubClass', 'LotArea', 'OverallQual'
    #         ,'OverallCond', 'YearBuilt', 'YearRemodAdd'
    #         ,'BsmtFinSF2'
    #         , 'BsmtUnfSF' 
    #         ,'TotalBsmtSF'
    #         , '1stFlrSF', '2ndFlrSF'
    #         ,'LowQualFinSF'
    #         , 'GrLivArea' 
    #         ,'BsmtFullBath' 
    #         ,'BsmtHalfBath'
    #         , 'FullBath'
    #         ,'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd'
    #         ,'Fireplaces' 
    #         ,'GarageCars'
    #         , 'GarageArea'
    #         , 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea'
    #         ,'MiscVal', 'MoSold', 'YrSold']
    # predictors = numPredNoNull
    
    # #All pre-processed attributes before Checkpoint 1
    # predictors = ['MSSubClass', 'LotArea', 'OverallQual','OverallCond', 'YearBuilt', 'YearRemodAdd'
    #         ,'BsmtFinSF2', 'BsmtUnfSF' ,'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea' 
    #         ,'BsmtFullBath' ,'BsmtHalfBath', 'FullBath','HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd'
    #         ,'Fireplaces' ,'GarageCars', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
    #         'ScreenPorch', 'PoolArea','MiscVal', 'MoSold', 'YrSold', 'CentralAir', 'LotShape','Neighborhood', 'BldgType', 
    #         'LandContour', 'Condition1', 'Condition2', 'Heating', 'HeatingQC', 'Functional', 'HouseStyle', 'ExterQual',
    #         'ExterCond']
    
    # ===============================================================================
    #                           ATTRIBUTES FROM EXPERIMENT B
    # ===============================================================================
    
    #All 78 pre-processed attributes
    # predictors = ['MSSubClass', 'LotArea', 'OverallQual','OverallCond', 'YearBuilt', 'YearRemodAdd'
    #         ,'BsmtFinSF2', 'BsmtUnfSF' ,'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea' 
    #         ,'BsmtFullBath' ,'BsmtHalfBath', 'FullBath','HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd'
    #         ,'Fireplaces' ,'GarageCars', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch'
    #         ,'ScreenPorch', 'PoolArea','MiscVal', 'MoSold', 'YrSold', 'CentralAir', 'LotShape','Neighborhood', 'BldgType'
    #         ,'LandContour', 'Condition1', 'Condition2', 'Heating', 'HeatingQC', 'Functional', 'HouseStyle', 'ExterQual'
    #         ,'ExterCond', 'MSZoning', 'LotFrontage', 'Street', 'Alley', 'Utilities', 'LotConfig', 'LandSlope', 'RoofStyle'
    #         ,'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'BsmtQual', 'BsmtCond'
    #         ,'BsmtExposure', 'BsmtFinSF1', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'KitchenQual', 'FireplaceQu'
    #         ,'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence'
    #         , 'MiscFeature', 'SaleType', 'SaleCondition']
    
    #The attributes found using correlation between numerical attributes and SalePrice attribute
    vipPredictors = ['OverallQual', 'GrLivArea', 'YearBuilt', 'YearRemodAdd', 
                      'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
                      'GarageCars', 'LotFrontage', 'MasVnrArea', 'BsmtFinSF1', '2ndFlrSF', 
                      'Fireplaces', 'GarageYrBlt', 'WoodDeckSF', 'OpenPorchSF',
                      'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'KitchenQual',
                      'Fence', 'SaleType', 'CentralAir', 'BldgType', 'LandContour', 'Condition1', 'HeatingQC',
                      'Functional', 'ExterQual']
                      #, 'Neighborhood']
    predictors =  vipPredictors
    
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    
    # ===============================================================================
    #                           PRE-PROCESSING NUMERIC ATTRIBUTES
    # ===============================================================================
    trainDF.loc[:, 'BsmtFinSF2'] = trainDF.loc[:, 'BsmtFinSF2'].fillna(0)
    testDF.loc[:, 'BsmtFinSF2'] = testDF.loc[:, 'BsmtFinSF2'].fillna(0)
    
    trainDF.loc[:, 'BsmtUnfSF'] = trainDF.loc[:, 'BsmtUnfSF'].fillna(trainDF.loc[:, 'BsmtUnfSF'].mode().iloc[0])
    testDF.loc[:, 'BsmtUnfSF'] = testDF.loc[:, 'BsmtUnfSF'].fillna(trainDF.loc[:, 'BsmtUnfSF'].mode().iloc[0])
    
    trainDF.loc[:, 'TotalBsmtSF'] = trainDF.loc[:, 'TotalBsmtSF'].fillna(0)
    testDF.loc[:, 'TotalBsmtSF'] = testDF.loc[:, 'TotalBsmtSF'].fillna(0)
    
    trainDF.loc[:, 'LowQualFinSF'] = trainDF.loc[:, 'LowQualFinSF'].fillna(0)
    testDF.loc[:, 'LowQualFinSF'] = testDF.loc[:, 'LowQualFinSF'].fillna(0)
    
    trainDF.loc[:, 'BsmtFullBath'] = trainDF.loc[:, 'BsmtFullBath'].fillna(0)
    testDF.loc[:, 'BsmtFullBath'] = testDF.loc[:, 'BsmtFullBath'].fillna(0)
    
    trainDF.loc[:, 'BsmtHalfBath'] = trainDF.loc[:, 'BsmtHalfBath'].fillna(0)
    testDF.loc[:, 'BsmtHalfBath'] = testDF.loc[:, 'BsmtHalfBath'].fillna(0)
    
    trainDF.loc[:, 'GarageCars'] = trainDF.loc[:, 'GarageCars'].fillna(0)
    testDF.loc[:, 'GarageCars'] = testDF.loc[:, 'GarageCars'].fillna(0)
    
    trainDF.loc[:, 'GarageArea'] = trainDF.loc[:, 'GarageArea'].fillna(0)
    testDF.loc[:, 'GarageArea'] = testDF.loc[:, 'GarageArea'].fillna(0)
    
    trainDF.loc[:, 'LotFrontage'] = trainDF.loc[:, 'LotFrontage'].fillna(trainDF.loc[:, 'LotFrontage'].mode().iloc[0])
    testDF.loc[:, 'LotFrontage'] = testDF.loc[:, 'LotFrontage'].fillna(trainDF.loc[:, 'LotFrontage'].mode().iloc[0])
    
    trainDF.loc[:, 'GarageYrBlt'] = trainDF.loc[:, 'GarageYrBlt'].fillna(trainDF.loc[:, 'GarageYrBlt'].mode().iloc[0])
    testDF.loc[:, 'GarageYrBlt'] = testDF.loc[:, 'GarageYrBlt'].fillna(trainDF.loc[:, 'GarageYrBlt'].mode().iloc[0])
   
    trainDF.loc[:, 'MasVnrArea'] = trainDF.loc[:, 'MasVnrArea'].fillna(trainDF.loc[:, 'MasVnrArea'].mode().iloc[0])
    testDF.loc[:, 'MasVnrArea'] = testDF.loc[:, 'MasVnrArea'].fillna(trainDF.loc[:, 'MasVnrArea'].mode().iloc[0])
    
    trainDF.loc[:, 'BsmtFinSF1'] = trainDF.loc[:, 'BsmtFinSF1'].fillna(trainDF.loc[:, 'BsmtFinSF1'].mode().iloc[0])
    testDF.loc[:, 'BsmtFinSF1'] = testDF.loc[:, 'BsmtFinSF1'].fillna(trainDF.loc[:, 'BsmtFinSF1'].mode().iloc[0])
  
    # ===============================================================================
    #                           PRE-PROCESSING NON-NUMERIC ATTRIBUTES
    # ===============================================================================
   
    #CENTRAl AIR VALUES: NaN, Y, N
    #PREPROCESSING METHOD: NaN = Mode(train), Y=1, N=0
    trainDF.loc[:, 'CentralAir'] = trainDF.loc[:, 'CentralAir'].map(lambda v: 0 if v=="N" else v)
    trainDF.loc[:, 'CentralAir'] = trainDF.loc[:, 'CentralAir'].map(lambda v: 1 if v=="Y" else v)
    trainDF.loc[:, 'CentralAir'] = trainDF.loc[:, 'CentralAir'].fillna(trainDF.loc[:, 'CentralAir'].mode().iloc[0])
    
    testDF.loc[:, 'CentralAir'] = testDF.loc[:, 'CentralAir'].map(lambda v: 0 if v=="N" else v)
    testDF.loc[:, 'CentralAir'] = testDF.loc[:, 'CentralAir'].map(lambda v: 1 if v=="Y" else v)
    testDF.loc[:, 'CentralAir'] = testDF.loc[:, 'CentralAir'].fillna(trainDF.loc[:, 'CentralAir'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #LOT SHAPE VALUES: NaN, 'Reg', 'IR1', 'IR2', 'IR3'
    #PREPROCESSING METHOD: NaN = Mode(train), Reg = 0, IR1 = 1, IR2 = 2, IR3 = 3
    trainDF.loc[:, 'LotShape'] = trainDF.loc[:, 'LotShape'].map(lambda v: 0 if v=="Reg" else v)
    trainDF.loc[:, 'LotShape'] = trainDF.loc[:, 'LotShape'].map(lambda v: 1 if v=="IR1" else v)
    trainDF.loc[:, 'LotShape'] = trainDF.loc[:, 'LotShape'].map(lambda v: 2 if v=="IR2" else v)
    trainDF.loc[:, 'LotShape'] = trainDF.loc[:, 'LotShape'].map(lambda v: 3 if v=="IR3" else v)
    trainDF.loc[:, 'LotShape'] = trainDF.loc[:, 'LotShape'].fillna(trainDF.loc[:, 'LotShape'].mode().iloc[0])
    
    testDF.loc[:, 'LotShape'] = testDF.loc[:, 'LotShape'].map(lambda v: 0 if v=="Reg" else v)
    testDF.loc[:, 'LotShape'] = testDF.loc[:, 'LotShape'].map(lambda v: 1 if v=="IR1" else v)
    testDF.loc[:, 'LotShape'] = testDF.loc[:, 'LotShape'].map(lambda v: 2 if v=="IR2" else v)
    testDF.loc[:, 'LotShape'] = testDF.loc[:, 'LotShape'].map(lambda v: 3 if v=="IR3" else v)
    testDF.loc[:, 'LotShape'] = testDF.loc[:, 'LotShape'].fillna(trainDF.loc[:, 'LotShape'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #NEIGHBORHOOD VALUES: NaN, 'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 
    #                       'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', SawyerW', 'IDOTRR', 'MeadowV', 
    #                       'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 
    #                       'SWISU', 'Blueste'
    #PREPROCESSING METHOD: NaN = Mode(train), [0-24]
    
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 0 if v=='CollgCr' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 1 if v=='Veenker' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 2 if v=='Crawfor' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 3 if v=='NoRidge' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 4 if v=='Mitchel' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 5 if v=='Somerst' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 6 if v=='NWAmes' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 7 if v=='OldTown' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 8 if v=='BrkSide' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 9 if v=='Sawyer' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 10 if v=="NridgHt" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 11 if v=="NAmes" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 12 if v=="SawyerW" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 13 if v=="IDOTRR" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 14 if v=="MeadowV" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 15 if v=="Edwards" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 16 if v=="Timber" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 17 if v=="Gilbert" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 18 if v=="StoneBr" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 19 if v=="ClearCr" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 20 if v=="NPkVill" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 21 if v=="Blmngtn" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 22 if v=="BrDale" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 23 if v=="SWISU" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 24 if v=="Blueste" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].fillna(trainDF.loc[:, 'Neighborhood'].mode().iloc[0])
    
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 0 if v=='CollgCr' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 1 if v=='Veenker' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 2 if v=='Crawfor' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 3 if v=='NoRidge' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 4 if v=='Mitchel' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 5 if v=='Somerst' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 6 if v=='NWAmes' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 7 if v=='OldTown' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 8 if v=='BrkSide' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 9 if v=='Sawyer' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 10 if v=="NridgHt" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 11 if v=="NAmes" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 12 if v=="SawyerW" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 13 if v=="IDOTRR" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 14 if v=="MeadowV" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 15 if v=="Edwards" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 16 if v=="Timber" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 17 if v=="Gilbert" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 18 if v=="StoneBr" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 19 if v=="ClearCr" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 20 if v=="NPkVill" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 21 if v=="Blmngtn" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 22 if v=="BrDale" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 23 if v=="SWISU" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 24 if v=="Blueste" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].fillna(trainDF.loc[:, 'Neighborhood'].mode().iloc[0])
   
        #----------------------------------------------------------------------------------------------------
    
    #BLDGTYPE VALUES: NaN,'1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'
    #PREPROCESSING METHOD: NaN = Mode(train), [0-4]
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].map(lambda v: 0 if v=="1Fam" else v)
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].map(lambda v: 1 if v=="2fmCon" else v)
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].map(lambda v: 2 if v=="Duplex" else v)
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].map(lambda v: 3 if v=="TwnhsE" else v)
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].map(lambda v: 4 if v=="Twnhs" else v)
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].fillna(trainDF.loc[:, 'BldgType'].mode().iloc[0])
    
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].map(lambda v: 0 if v=="1Fam" else v)
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].map(lambda v: 1 if v=="2fmCon" else v)
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].map(lambda v: 2 if v=="Duplex" else v)
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].map(lambda v: 3 if v=="TwnhsE" else v)
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].map(lambda v: 4 if v=="Twnhs" else v)
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].fillna(trainDF.loc[:, 'BldgType'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #LAND CONTOUR VALUES: NaN,'Lvl', 'Bnk', 'Low', 'HLS'
    #PREPROCESSING METHOD: NaN = Mode(train), [0-3]
    trainDF.loc[:, 'LandContour'] = trainDF.loc[:, 'LandContour'].map(lambda v: 0 if v=="Lvl" else v)
    trainDF.loc[:, 'LandContour'] = trainDF.loc[:, 'LandContour'].map(lambda v: 1 if v=="Bnk" else v)
    trainDF.loc[:, 'LandContour'] = trainDF.loc[:, 'LandContour'].map(lambda v: 2 if v=="Low" else v)
    trainDF.loc[:, 'LandContour'] = trainDF.loc[:, 'LandContour'].map(lambda v: 3 if v=="HLS" else v)
    trainDF.loc[:, 'LandContour'] = trainDF.loc[:, 'LandContour'].fillna(trainDF.loc[:, 'LandContour'].mode().iloc[0])
    
    testDF.loc[:, 'LandContour'] = testDF.loc[:, 'LandContour'].map(lambda v: 0 if v=="Lvl" else v)
    testDF.loc[:, 'LandContour'] = testDF.loc[:, 'LandContour'].map(lambda v: 1 if v=="Bnk" else v)
    testDF.loc[:, 'LandContour'] = testDF.loc[:, 'LandContour'].map(lambda v: 2 if v=="Low" else v)
    testDF.loc[:, 'LandContour'] = testDF.loc[:, 'LandContour'].map(lambda v: 3 if v=="HLS" else v)
    testDF.loc[:, 'LandContour'] = testDF.loc[:, 'LandContour'].fillna(trainDF.loc[:, 'LandContour'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #CONDITION 1 VALUES: NaN, 'Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe'
    #PREPROCESSING METHOD: NaN = Mode(train), 'Norm' = 0, 'Feedr' = 1, 'PosN' = 2, 'Artery' = 3, 'RRAe' = 4, 
    #                       'RRNn' = 5, 'RRAn' = 6, 'PosA' = 7, 'RRNe' = 8
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 0 if v=="Norm" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 1 if v=="Feedr" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 2 if v=="PosN" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 3 if v=="Artery" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 4 if v=="RRAe" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 5 if v=="RRNn" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 6 if v=="RRAn" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 7 if v=="PosA" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 8 if v=="RRNe" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].fillna(trainDF.loc[:, 'Condition1'].mode().iloc[0])
    
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 0 if v=="Norm" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 1 if v=="Feedr" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 2 if v=="PosN" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 3 if v=="Artery" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 4 if v=="RRAe" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 5 if v=="RRNn" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 6 if v=="RRAn" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 7 if v=="PosA" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 8 if v=="RRNe" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].fillna(trainDF.loc[:, 'Condition1'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #CONDITION 2 VALUES: NaN, 'Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe'
    #PREPROCESSING METHOD: NaN = Mode(train), 'Norm' = 0, 'Feedr' = 1, 'PosN' = 2, 'Artery' = 3, 'RRAe' = 4, 
    #                       'RRNn' = 5, 'RRAn' = 6, 'PosA' = 7
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 0 if v=="Norm" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 1 if v=="Feedr" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 2 if v=="PosN" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 3 if v=="Artery" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 4 if v=="RRAe" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 5 if v=="RRNn" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 6 if v=="RRAn" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 7 if v=="PosA" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].fillna(trainDF.loc[:, 'Condition2'].mode().iloc[0])
    
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 0 if v=="Norm" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 1 if v=="Feedr" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 2 if v=="PosN" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 3 if v=="Artery" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 4 if v=="RRAe" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 5 if v=="RRNn" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 6 if v=="RRAn" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 7 if v=="PosA" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].fillna(trainDF.loc[:, 'Condition2'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #HEATING VALUES: NaN,'GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'
    #PREPROCESSING METHOD: NaN = Mode(train), 'GasA' = 0, 'GasW' = 1, 'Grav' = 2, 'Wall' = 3, 'OthW' = 4, 'Floor' = 5
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 0 if v=="GasA" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 1 if v=="GasW" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 2 if v=="Grav" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 3 if v=="Wall" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 4 if v=="OthW" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 5 if v=="Floor" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].fillna(trainDF.loc[:, 'Heating'].mode().iloc[0])
    
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 0 if v=="GasA" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 1 if v=="GasW" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 2 if v=="Grav" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 3 if v=="Wall" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 4 if v=="OthW" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 5 if v=="Floor" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].fillna(trainDF.loc[:, 'Heating'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #HEATING QC VALUES: NaN,'Ex', 'Gd', 'TA', 'Fa', 'Po'
    #PREPROCESSING METHOD: NaN = Median(train), 'Ex' = 0, 'Gd' = 1, 'TA' = 2, 'Fa' = 3, 'Po' = 4
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].map(lambda v: 4 if v=="Ex" else v)
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].map(lambda v: 3 if v=="Gd" else v)
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].map(lambda v: 2 if v=="TA" else v)
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].map(lambda v: 1 if v=="Fa" else v)
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].map(lambda v: 0 if v=="Po" else v)
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].fillna(trainDF.loc[:, 'HeatingQC'].median())
    
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].map(lambda v: 4 if v=="Ex" else v)
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].map(lambda v: 3 if v=="Gd" else v)
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].map(lambda v: 1 if v=="Fa" else v)
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].map(lambda v: 0 if v=="Po" else v)
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].fillna(trainDF.loc[:, 'HeatingQC'].median())
    
        #----------------------------------------------------------------------------------------------------
    
    #FUNCTIONAL VALUES: NaN,'Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev'
    #PREPROCESSING METHOD: NaN = Median(train), 'Typ' = 0, 'Min1' = 1, 'Maj1' = 2, 'Min2' = 3, 'Mod' = 4, 'Maj2' = 5, 'Sev' = 6
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 0 if v=="Typ" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 1 if v=="Min1" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 2 if v=="Maj1" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 3 if v=="Min2" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 4 if v=="Mod" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 5 if v=="Maj2" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 6 if v=="Sev" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].fillna(trainDF.loc[:, 'Functional'].mode().iloc[0])
    
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 0 if v=="Typ" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 1 if v=="Min1" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 2 if v=="Maj1" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 3 if v=="Min2" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 4 if v=="Mod" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 5 if v=="Maj2" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 6 if v=="Sev" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].fillna(trainDF.loc[:, 'Functional'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------

    #HOUSE STYLE VALUES: NaN, '2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'
    #PREPROCESSING METHOD: NaN = Median(train), '2Story' = 0, '1Story' = 1, '1.5Fin' = 2, '1.5Unf' = 3, 'SFoyer' = 4, 
    #                       'SLvl' = 5, '2.5Unf' = 6, '2.5Fin' = 7
    trainDF.loc[:, 'HouseStyle'] = trainDF.loc[:, 'HouseStyle'].map(lambda v: 0 if v=="2Story" else v)
    trainDF.loc[:, 'HouseStyle'] = trainDF.loc[:, 'HouseStyle'].map(lambda v: 1 if v=="1Story" else v)
    trainDF.loc[:, 'HouseStyle'] = trainDF.loc[:, 'HouseStyle'].map(lambda v: 2 if v=="1.5Fin" else v)
    trainDF.loc[:, 'HouseStyle'] = trainDF.loc[:, 'HouseStyle'].map(lambda v: 3 if v=="1.5Unf" else v)
    trainDF.loc[:, 'HouseStyle'] = trainDF.loc[:, 'HouseStyle'].map(lambda v: 4 if v=="SFoyer" else v)
    trainDF.loc[:, 'HouseStyle'] = trainDF.loc[:, 'HouseStyle'].map(lambda v: 5 if v=="SLvl" else v)
    trainDF.loc[:, 'HouseStyle'] = trainDF.loc[:, 'HouseStyle'].map(lambda v: 6 if v=="2.5Unf" else v)
    trainDF.loc[:, 'HouseStyle'] = trainDF.loc[:, 'HouseStyle'].map(lambda v: 7 if v=="2.5Fin" else v)
    trainDF.loc[:, 'HouseStyle'] = trainDF.loc[:, 'HouseStyle'].fillna(trainDF.loc[:, 'HouseStyle'].median())
    
    testDF.loc[:, 'HouseStyle'] = testDF.loc[:, 'HouseStyle'].map(lambda v: 0 if v=="2Story" else v)
    testDF.loc[:, 'HouseStyle'] = testDF.loc[:, 'HouseStyle'].map(lambda v: 1 if v=="1Story" else v)
    testDF.loc[:, 'HouseStyle'] = testDF.loc[:, 'HouseStyle'].map(lambda v: 2 if v=="1.5Fin" else v)
    testDF.loc[:, 'HouseStyle'] = testDF.loc[:, 'HouseStyle'].map(lambda v: 3 if v=="1.5Unf" else v)
    testDF.loc[:, 'HouseStyle'] = testDF.loc[:, 'HouseStyle'].map(lambda v: 4 if v=="SFoyer" else v)
    testDF.loc[:, 'HouseStyle'] = testDF.loc[:, 'HouseStyle'].map(lambda v: 5 if v=="SLvl" else v)
    testDF.loc[:, 'HouseStyle'] = testDF.loc[:, 'HouseStyle'].map(lambda v: 6 if v=="2.5Unf" else v)
    testDF.loc[:, 'HouseStyle'] = testDF.loc[:, 'HouseStyle'].map(lambda v: 7 if v=="2.5Fin" else v)
    testDF.loc[:, 'HouseStyle'] = testDF.loc[:, 'HouseStyle'].fillna(trainDF.loc[:, 'HouseStyle'].median())
    
        #----------------------------------------------------------------------------------------------------
    
    #EXTER QUAL VALUES: NaN, 'Gd', 'TA', 'Ex', 'Fa'
    #PREPROCESSING METHOD: NaN = Mode(train), Gd = 0, TA = 1, Ex = 2, Fa = 3 [Ex=0, Gd=1, Ta=2, Fa=3]
    trainDF.loc[:, 'ExterQual'] = trainDF.loc[:, 'ExterQual'].map(lambda v: 3 if v=="Ex" else v)
    trainDF.loc[:, 'ExterQual'] = trainDF.loc[:, 'ExterQual'].map(lambda v: 2 if v=="Gd" else v)
    trainDF.loc[:, 'ExterQual'] = trainDF.loc[:, 'ExterQual'].map(lambda v: 1 if v=="TA" else v)
    trainDF.loc[:, 'ExterQual'] = trainDF.loc[:, 'ExterQual'].map(lambda v: 0 if v=="Fa" else v)
    trainDF.loc[:, 'ExterQual'] = trainDF.loc[:, 'ExterQual'].fillna(trainDF.loc[:, 'ExterQual'].mode().iloc[0])
    
    testDF.loc[:, 'ExterQual'] = testDF.loc[:, 'ExterQual'].map(lambda v: 3 if v=="Ex" else v)
    testDF.loc[:, 'ExterQual'] = testDF.loc[:, 'ExterQual'].map(lambda v: 2 if v=="Gd" else v)
    testDF.loc[:, 'ExterQual'] = testDF.loc[:, 'ExterQual'].map(lambda v: 1 if v=="TA" else v)
    testDF.loc[:, 'ExterQual'] = testDF.loc[:, 'ExterQual'].map(lambda v: 0 if v=="Fa" else v)
    testDF.loc[:, 'ExterQual'] = testDF.loc[:, 'ExterQual'].fillna(trainDF.loc[:, 'ExterQual'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------

    #EXTER COND VALUES: NaN, 'TA', 'Gd', 'Fa', 'Po', 'Ex'
    #PREPROCESSING METHOD: NaN = Mode(train), Gd = 0, TA = 1, Ex = 2, Fa = 3, Po = 4
    trainDF.loc[:, 'ExterCond'] = trainDF.loc[:, 'ExterCond'].map(lambda v: 4 if v=="Ex" else v)
    trainDF.loc[:, 'ExterCond'] = trainDF.loc[:, 'ExterCond'].map(lambda v: 3 if v=="Gd" else v)
    trainDF.loc[:, 'ExterCond'] = trainDF.loc[:, 'ExterCond'].map(lambda v: 2 if v=="TA" else v)
    trainDF.loc[:, 'ExterCond'] = trainDF.loc[:, 'ExterCond'].map(lambda v: 1 if v=="Fa" else v)
    trainDF.loc[:, 'ExterCond'] = trainDF.loc[:, 'ExterCond'].map(lambda v: 0 if v=="Po" else v)
    trainDF.loc[:, 'ExterCond'] = trainDF.loc[:, 'ExterCond'].fillna(trainDF.loc[:, 'ExterCond'].mode().iloc[0])
    
    testDF.loc[:, 'ExterCond'] = testDF.loc[:, 'ExterCond'].map(lambda v: 4 if v=="Ex" else v)
    testDF.loc[:, 'ExterCond'] = testDF.loc[:, 'ExterCond'].map(lambda v: 3 if v=="Gd" else v)
    testDF.loc[:, 'ExterCond'] = testDF.loc[:, 'ExterCond'].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:, 'ExterCond'] = testDF.loc[:, 'ExterCond'].map(lambda v: 1 if v=="Fa" else v)
    testDF.loc[:, 'ExterCond'] = testDF.loc[:, 'ExterCond'].map(lambda v: 0 if v=="Po" else v)
    testDF.loc[:, 'ExterCond'] = testDF.loc[:, 'ExterCond'].fillna(trainDF.loc[:, 'ExterCond'].mode().iloc[0])
   
        #----------------------------------------------------------------------------------------------------
    
    #MSZoning values: NaN, 'RL', 'RM', 'C (all)', 'FV', 'RH'
    #Preprocessing method: NaN = Mode(trainDf), 'RL'=0,'RM'=1, 'C (all)'=2, 'FV'=3,'RH'=4
    trainDF.loc[:, 'MSZoning'] = trainDF.loc[:, 'MSZoning'].map(lambda v: 0 if v=="RL" else v)
    trainDF.loc[:, 'MSZoning'] = trainDF.loc[:, 'MSZoning'].map(lambda v: 1 if v=="RM" else v)
    trainDF.loc[:, 'MSZoning'] = trainDF.loc[:, 'MSZoning'].map(lambda v: 2 if v=="C (all)" else v)
    trainDF.loc[:, 'MSZoning'] = trainDF.loc[:, 'MSZoning'].map(lambda v: 3 if v=="FV" else v)
    trainDF.loc[:, 'MSZoning'] = trainDF.loc[:, 'MSZoning'].map(lambda v: 4 if v=="RH" else v)
    trainDF.loc[:, 'MSZoning'] = trainDF.loc[:, 'MSZoning'].fillna(trainDF.loc[:, 'MSZoning'].mode().iloc[0])
    
    testDF.loc[:, 'MSZoning'] = testDF.loc[:, 'MSZoning'].map(lambda v: 0 if v=="RL" else v)
    testDF.loc[:, 'MSZoning'] = testDF.loc[:, 'MSZoning'].map(lambda v: 1 if v=="RM" else v)
    testDF.loc[:, 'MSZoning'] = testDF.loc[:, 'MSZoning'].map(lambda v: 2 if v=="C (all)" else v)
    testDF.loc[:, 'MSZoning'] = testDF.loc[:, 'MSZoning'].map(lambda v: 3 if v=="FV" else v)
    testDF.loc[:, 'MSZoning'] = testDF.loc[:, 'MSZoning'].map(lambda v: 4 if v=="RH" else v)
    testDF.loc[:, 'MSZoning'] = testDF.loc[:, 'MSZoning'].fillna(trainDF.loc[:, 'MSZoning'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #Street Values: NaN, 'Pave', 'Grvl'
    #Preprocessing Method: NaN = Mode(trainDf), 'Pave'=0,'Grvl'=1
    trainDF.loc[:, 'Street'] = trainDF.loc[:, 'Street'].map(lambda v: 0 if v=="Pave" else v)
    trainDF.loc[:, 'Street'] = trainDF.loc[:, 'Street'].map(lambda v: 1 if v=="Grvl" else v)
    trainDF.loc[:, 'Street'] = trainDF.loc[:, 'Street'].fillna(trainDF.loc[:, 'Street'].mode().iloc[0])
    
    testDF.loc[:, 'Street'] = testDF.loc[:, 'Street'].map(lambda v: 0 if v=="Pave" else v)
    testDF.loc[:, 'Street'] = testDF.loc[:, 'Street'].map(lambda v: 1 if v=="Grvl" else v)
    testDF.loc[:, 'Street'] = testDF.loc[:, 'Street'].fillna(trainDF.loc[:, 'Street'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #Alley Values: NaN, 'NA', 'Grvl', 'Pave'
    #Preprocessing Method: NaN = Mode(trainDf), 'NA'=0, 'Pave'=1,'Grvl'=2
    trainDF.loc[:, 'Alley'] = trainDF.loc[:, 'Alley'].map(lambda v: 0 if v=="NA" else v)
    trainDF.loc[:, 'Alley'] = trainDF.loc[:, 'Alley'].map(lambda v: 1 if v=="Pave" else v)
    trainDF.loc[:, 'Alley'] = trainDF.loc[:, 'Alley'].map(lambda v: 2 if v=="Grvl" else v)
    trainDF.loc[:, 'Alley'] = trainDF.loc[:, 'Alley'].fillna(trainDF.loc[:, 'Alley'].mode().iloc[0])
    
    testDF.loc[:, 'Alley'] = testDF.loc[:, 'Alley'].map(lambda v: 0 if v=="NA" else v)
    testDF.loc[:, 'Alley'] = testDF.loc[:, 'Alley'].map(lambda v: 1 if v=="Pave" else v)
    testDF.loc[:, 'Alley'] = testDF.loc[:, 'Alley'].map(lambda v: 2 if v=="Grvl" else v)
    testDF.loc[:, 'Alley'] = testDF.loc[:, 'Alley'].fillna(trainDF.loc[:, 'Alley'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #Utilities Values: NaN, 'AllPub', 'NoSeWa'
    #Preprocessing Method: NaN = Mode(trainDf), 'AllPub'=0,'NoSeWa'=1
    trainDF.loc[:, 'Utilities'] = trainDF.loc[:, 'Utilities'].map(lambda v: 0 if v=="AllPub" else v)
    trainDF.loc[:, 'Utilities'] = trainDF.loc[:, 'Utilities'].map(lambda v: 1 if v=="NoSeWa" else v)
    trainDF.loc[:, 'Utilities'] = trainDF.loc[:, 'Utilities'].fillna(trainDF.loc[:, 'Utilities'].mode().iloc[0])
    
    testDF.loc[:, 'Utilities'] = testDF.loc[:, 'Utilities'].map(lambda v: 0 if v=="AllPub" else v)
    testDF.loc[:, 'Utilities'] = testDF.loc[:, 'Utilities'].map(lambda v: 1 if v=="NoSeWa" else v)
    testDF.loc[:, 'Utilities'] = testDF.loc[:, 'Utilities'].fillna(trainDF.loc[:, 'Utilities'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #LotConfig Values: NaN, 'Inside', 'FR2', 'Corner', 'CulDSac', 'FR3'
    #Preprocessing Method: NaN = Mode(trainDf), 'Inside'=0, 'FR2'=1, 'Corner'=2, 'CulDSac'=3, 'FR3'=4
    trainDF.loc[:, 'LotConfig'] = trainDF.loc[:, 'LotConfig'].map(lambda v: 0 if v=="Inside" else v)
    trainDF.loc[:, 'LotConfig'] = trainDF.loc[:, 'LotConfig'].map(lambda v: 1 if v=="FR2" else v)
    trainDF.loc[:, 'LotConfig'] = trainDF.loc[:, 'LotConfig'].map(lambda v: 2 if v=="Corner" else v)
    trainDF.loc[:, 'LotConfig'] = trainDF.loc[:, 'LotConfig'].map(lambda v: 3 if v=="CulDSac" else v)
    trainDF.loc[:, 'LotConfig'] = trainDF.loc[:, 'LotConfig'].map(lambda v: 4 if v=="FR3" else v)
    trainDF.loc[:, 'LotConfig'] = trainDF.loc[:, 'LotConfig'].fillna(trainDF.loc[:, 'LotConfig'].mode().iloc[0])
    
    testDF.loc[:, 'LotConfig'] = testDF.loc[:, 'LotConfig'].map(lambda v: 0 if v=="Inside" else v)
    testDF.loc[:, 'LotConfig'] = testDF.loc[:, 'LotConfig'].map(lambda v: 1 if v=="FR2" else v)
    testDF.loc[:, 'LotConfig'] = testDF.loc[:, 'LotConfig'].map(lambda v: 2 if v=="Corner" else v)
    testDF.loc[:, 'LotConfig'] = testDF.loc[:, 'LotConfig'].map(lambda v: 3 if v=="CulDSac" else v)
    testDF.loc[:, 'LotConfig'] = testDF.loc[:, 'LotConfig'].map(lambda v: 4 if v=="FR3" else v)
    testDF.loc[:, 'LotConfig'] = testDF.loc[:, 'LotConfig'].fillna(trainDF.loc[:, 'LotConfig'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #LandSlope Values: NaN, 'Gtl', 'Mod', 'Sev'
    #Preprocessing Method: NaN = Mode(trainDf), 'Gtl'=0, 'Mod'=1,'Sev'=2
    trainDF.loc[:, 'LandSlope'] = trainDF.loc[:, 'LandSlope'].map(lambda v: 0 if v=="Gtl" else v)
    trainDF.loc[:, 'LandSlope'] = trainDF.loc[:, 'LandSlope'].map(lambda v: 1 if v=="Mod" else v)
    trainDF.loc[:, 'LandSlope'] = trainDF.loc[:, 'LandSlope'].map(lambda v: 2 if v=="Sev" else v)
    trainDF.loc[:, 'LandSlope'] = trainDF.loc[:, 'LandSlope'].fillna(trainDF.loc[:, 'LandSlope'].mode().iloc[0])
    
    testDF.loc[:, 'LandSlope'] = testDF.loc[:, 'LandSlope'].map(lambda v: 0 if v=="Gtl" else v)
    testDF.loc[:, 'LandSlope'] = testDF.loc[:, 'LandSlope'].map(lambda v: 1 if v=="Mod" else v)
    testDF.loc[:, 'LandSlope'] = testDF.loc[:, 'LandSlope'].map(lambda v: 2 if v=="Sev" else v)
    testDF.loc[:, 'LandSlope'] = testDF.loc[:, 'LandSlope'].fillna(trainDF.loc[:, 'LandSlope'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #RoofStyle Values: NaN, 'Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'
    #Preprocessing Method: NaN = Mode(trainDf), 'Gable'=0, 'Hip'=1, 'Gambrel'=2, 'Mansard'=3, 'Flat'=4, 'Shed'=5

    trainDF.loc[:, 'RoofStyle'] = trainDF.loc[:, 'RoofStyle'].map(lambda v: 0 if v=="Gable" else v)
    trainDF.loc[:, 'RoofStyle'] = trainDF.loc[:, 'RoofStyle'].map(lambda v: 1 if v=="Hip" else v)
    trainDF.loc[:, 'RoofStyle'] = trainDF.loc[:, 'RoofStyle'].map(lambda v: 2 if v=="Gambrel" else v)
    trainDF.loc[:, 'RoofStyle'] = trainDF.loc[:, 'RoofStyle'].map(lambda v: 3 if v=="Mansard" else v)
    trainDF.loc[:, 'RoofStyle'] = trainDF.loc[:, 'RoofStyle'].map(lambda v: 4 if v=="Flat" else v)
    trainDF.loc[:, 'RoofStyle'] = trainDF.loc[:, 'RoofStyle'].map(lambda v: 5 if v=="Shed" else v)
    trainDF.loc[:, 'RoofStyle'] = trainDF.loc[:, 'RoofStyle'].fillna(trainDF.loc[:, 'RoofStyle'].mode().iloc[0])
    
    testDF.loc[:, 'RoofStyle'] = testDF.loc[:, 'RoofStyle'].map(lambda v: 0 if v=="Gable" else v)
    testDF.loc[:, 'RoofStyle'] = testDF.loc[:, 'RoofStyle'].map(lambda v: 1 if v=="Hip" else v)
    testDF.loc[:, 'RoofStyle'] = testDF.loc[:, 'RoofStyle'].map(lambda v: 2 if v=="Gambrel" else v)
    testDF.loc[:, 'RoofStyle'] = testDF.loc[:, 'RoofStyle'].map(lambda v: 3 if v=="Mansard" else v)
    testDF.loc[:, 'RoofStyle'] = testDF.loc[:, 'RoofStyle'].map(lambda v: 4 if v=="Flat" else v)
    testDF.loc[:, 'RoofStyle'] = testDF.loc[:, 'RoofStyle'].map(lambda v: 5 if v=="Shed" else v)
    testDF.loc[:, 'RoofStyle'] = testDF.loc[:, 'RoofStyle'].fillna(trainDF.loc[:, 'RoofStyle'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #RoofMatl Values: NaN, 'CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv', 'Roll', 'ClyTile'
    #Preprocessing Method: NaN = Mode(trainDf), 'CompShg'=0, 'WdShngl'=1, 'Metal'=2, 'WdShake'=3, 'Membran'=4, 
    #                       'Tar&Grv'=5, 'Roll'=6, 'ClyTile'=7
    trainDF.loc[:, 'RoofMatl'] = trainDF.loc[:, 'RoofMatl'].map(lambda v: 0 if v=="CompShg" else v)
    trainDF.loc[:, 'RoofMatl'] = trainDF.loc[:, 'RoofMatl'].map(lambda v: 1 if v=="WdShngl" else v)
    trainDF.loc[:, 'RoofMatl'] = trainDF.loc[:, 'RoofMatl'].map(lambda v: 2 if v=="Metal" else v)
    trainDF.loc[:, 'RoofMatl'] = trainDF.loc[:, 'RoofMatl'].map(lambda v: 3 if v=="WdShake" else v)
    trainDF.loc[:, 'RoofMatl'] = trainDF.loc[:, 'RoofMatl'].map(lambda v: 4 if v=="Membran" else v)
    trainDF.loc[:, 'RoofMatl'] = trainDF.loc[:, 'RoofMatl'].map(lambda v: 5 if v=="Tar&Grv" else v)
    trainDF.loc[:, 'RoofMatl'] = trainDF.loc[:, 'RoofMatl'].map(lambda v: 6 if v=="Roll" else v)
    trainDF.loc[:, 'RoofMatl'] = trainDF.loc[:, 'RoofMatl'].map(lambda v: 7 if v=="ClyTile" else v)
    trainDF.loc[:, 'RoofMatl'] = trainDF.loc[:, 'RoofMatl'].fillna(trainDF.loc[:, 'RoofMatl'].mode().iloc[0])
    
    testDF.loc[:, 'RoofMatl'] = testDF.loc[:, 'RoofMatl'].map(lambda v: 0 if v=="CompShg" else v)
    testDF.loc[:, 'RoofMatl'] = testDF.loc[:, 'RoofMatl'].map(lambda v: 1 if v=="WdShngl" else v)
    testDF.loc[:, 'RoofMatl'] = testDF.loc[:, 'RoofMatl'].map(lambda v: 2 if v=="Metal" else v)
    testDF.loc[:, 'RoofMatl'] = testDF.loc[:, 'RoofMatl'].map(lambda v: 3 if v=="WdShake" else v)
    testDF.loc[:, 'RoofMatl'] = testDF.loc[:, 'RoofMatl'].map(lambda v: 4 if v=="Membran" else v)
    testDF.loc[:, 'RoofMatl'] = testDF.loc[:, 'RoofMatl'].map(lambda v: 5 if v=="Tar&Grv" else v)
    testDF.loc[:, 'RoofMatl'] = testDF.loc[:, 'RoofMatl'].map(lambda v: 6 if v=="Roll" else v)
    testDF.loc[:, 'RoofMatl'] = testDF.loc[:, 'RoofMatl'].map(lambda v: 7 if v=="ClyTile" else v)
    testDF.loc[:, 'RoofMatl'] = testDF.loc[:, 'RoofMatl'].fillna(trainDF.loc[:, 'RoofMatl'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #Exterior1st Values: NaN, 'VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 
    #                    'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock'
    #Preprocessing Method: NaN = Mode(trainDf), 'VinylSd'=0, 'MetalSd'=1, 'Wd Sdng'=2, 'HdBoard'=3, 'BrkFace'=4, 
    #                       'WdShing'=5, 'CemntBd'=6, 'Plywood'=7, 'AsbShng'=8, 'Stucco'=9, 'BrkComm'=10, 'AsphShn'=11, 
    #                       'Stone'=12, 'ImStucc'=13, 'CBlock'=14
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 0 if v=='VinylSd' else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 1 if v=='MetalSd' else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 2 if v=='Wd Sdng' else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 3 if v=='HdBoard' else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 4 if v=='BrkFace' else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 5 if v=='WdShing' else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 6 if v=='CemntBd' else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 7 if v=='Plywood' else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 8 if v=='AsbShng' else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 9 if v=='Stucco' else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 10 if v=="BrkComm" else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 11 if v=="AsphShn" else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 12 if v=="Stone" else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 13 if v=="ImStucc" else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].map(lambda v: 14 if v=="CBlock" else v)
    trainDF.loc[:, 'Exterior1st'] = trainDF.loc[:, 'Exterior1st'].fillna(trainDF.loc[:, 'Exterior1st'].mode().iloc[0])
    
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 0 if v=='VinylSd' else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 1 if v=='MetalSd' else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 2 if v=='Wd Sdng' else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 3 if v=='HdBoard' else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 4 if v=='BrkFace' else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 5 if v=='Wd Shng' else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 6 if v=='CemntBd' else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 7 if v=='Plywood' else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 8 if v=='AsbShng' else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 9 if v=='Stucco' else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 10 if v=="BrkComm" else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 11 if v=="AsphShn" else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 12 if v=="Stone" else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 13 if v=="ImStucc" else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].map(lambda v: 14 if v=="CBlock" else v)
    testDF.loc[:, 'Exterior1st'] = testDF.loc[:, 'Exterior1st'].fillna(trainDF.loc[:, 'Exterior1st'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #Exterior2nd Values: NaN,'VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng', 'CmentBd', 'BrkFace', 
    #                    'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc', 'AsphShn', 'Stone', 'Other', 'CBlock'
    #Preprocessing Method: NaN = Mode(trainDf), 'VinylSd'=0, 'MetalSd'=1, 'Wd Shng'=2, 'HdBoard'=3, 'Plywood'=4, 'Wd Sdng'=5, 
    #                        'CmentBd'=6, 'BrkFace'=7, 'Stucco'=8, 'AsbShng'=9, 'Brk Cmn'=10, 'ImStucc'=11, 'AsphShn'=12, 
    #                          'Stone'=13, 'Other'=14, 'CBlock'=15
   
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 0 if v=='AsbShng' else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 1 if v=='AsphShn' else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 2 if v=='Brk Cmn' else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 3 if v=='BrkFace' else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 4 if v=='CBlock' else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 5 if v=='CmentBd' else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 6 if v=='HdBoard' else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 7 if v=='ImStucc' else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 8 if v=='MetalSd' else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 9 if v=='Other' else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 10 if v=="Plywood" else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 11 if v=="PreCast" else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 12 if v=="Stone" else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 13 if v=="Stucco" else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 14 if v=="VinylSd" else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 15 if v=="Wd Sdng" else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].map(lambda v: 16 if v=="Wd Shng" else v)
    trainDF.loc[:, 'Exterior2nd'] = trainDF.loc[:, 'Exterior2nd'].fillna(trainDF.loc[:, 'Exterior2nd'].mode().iloc[0])
    
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 0 if v=='AsbShng' else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 1 if v=='AsphShn' else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 2 if v=='Brk Cmn' else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 3 if v=='BrkFace' else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 4 if v=='CBlock' else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 5 if v=='CmentBd' else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 6 if v=='HdBoard' else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 7 if v=='ImStucc' else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 8 if v=='MetalSd' else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 9 if v=='Other' else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 10 if v=="Plywood" else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 11 if v=="PreCast" else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 12 if v=="Stone" else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 13 if v=="Stucco" else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 14 if v=="VinylSd" else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 15 if v=="Wd Sdng" else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].map(lambda v: 16 if v=="Wd Shng" else v)
    testDF.loc[:, 'Exterior2nd'] = testDF.loc[:, 'Exterior2nd'].fillna(trainDF.loc[:, 'Exterior2nd'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    # MasVnrType Values: NaN, 'BrkFace', 'None', 'Stone', 'BrkCmn'
    # Preprocessing method: NaN = Mode(trainDf), 'BrkFace'=0, 'None'=1, 'Stone'=2, 'BrkCmn'=3
    trainDF.loc[:, 'MasVnrType'] = trainDF.loc[:, 'MasVnrType'].map(lambda v: 0 if v=="BrkFace" else v)
    trainDF.loc[:, 'MasVnrType'] = trainDF.loc[:, 'MasVnrType'].map(lambda v: 1 if v=="None" else v)
    trainDF.loc[:, 'MasVnrType'] = trainDF.loc[:, 'MasVnrType'].map(lambda v: 2 if v=="Stone" else v)
    trainDF.loc[:, 'MasVnrType'] = trainDF.loc[:, 'MasVnrType'].map(lambda v: 3 if v=="BrkCmn" else v)
    trainDF.loc[:, 'MasVnrType'] = trainDF.loc[:, 'MasVnrType'].fillna(trainDF.loc[:, 'MasVnrType'].mode().iloc[0])
    
    testDF.loc[:, 'MasVnrType'] = testDF.loc[:, 'MasVnrType'].map(lambda v: 0 if v=="BrkFace" else v)
    testDF.loc[:, 'MasVnrType'] = testDF.loc[:, 'MasVnrType'].map(lambda v: 1 if v=="None" else v)
    testDF.loc[:, 'MasVnrType'] = testDF.loc[:, 'MasVnrType'].map(lambda v: 2 if v=="Stone" else v)
    testDF.loc[:, 'MasVnrType'] = testDF.loc[:, 'MasVnrType'].map(lambda v: 3 if v=="BrkCmn" else v)
    testDF.loc[:, 'MasVnrType'] = testDF.loc[:, 'MasVnrType'].fillna(trainDF.loc[:, 'MasVnrType'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------

    #Foundation Values: NaN, 'PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'
    #Preprocessing method: NaN = Mode(trainDf), 'PConc'=0, 'CBlock'=1, 'BrkTil'=2, 'Wood'=3, 'Slab'=4, 'Stone'=5
    trainDF.loc[:, 'Foundation'] = trainDF.loc[:, 'Foundation'].map(lambda v: 0 if v=="PConc" else v)
    trainDF.loc[:, 'Foundation'] = trainDF.loc[:, 'Foundation'].map(lambda v: 1 if v=="CBlock" else v)
    trainDF.loc[:, 'Foundation'] = trainDF.loc[:, 'Foundation'].map(lambda v: 2 if v=="BrkTil" else v)
    trainDF.loc[:, 'Foundation'] = trainDF.loc[:, 'Foundation'].map(lambda v: 3 if v=="Wood" else v)
    trainDF.loc[:, 'Foundation'] = trainDF.loc[:, 'Foundation'].map(lambda v: 4 if v=="Slab" else v)
    trainDF.loc[:, 'Foundation'] = trainDF.loc[:, 'Foundation'].map(lambda v: 5 if v=="Stone" else v)
    trainDF.loc[:, 'Foundation'] = trainDF.loc[:, 'Foundation'].fillna(trainDF.loc[:, 'Foundation'].mode().iloc[0])
    
    testDF.loc[:, 'Foundation'] = testDF.loc[:, 'Foundation'].map(lambda v: 0 if v=="PConc" else v)
    testDF.loc[:, 'Foundation'] = testDF.loc[:, 'Foundation'].map(lambda v: 1 if v=="CBlock" else v)
    testDF.loc[:, 'Foundation'] = testDF.loc[:, 'Foundation'].map(lambda v: 2 if v=="BrkTil" else v)
    testDF.loc[:, 'Foundation'] = testDF.loc[:, 'Foundation'].map(lambda v: 3 if v=="Wood" else v)
    testDF.loc[:, 'Foundation'] = testDF.loc[:, 'Foundation'].map(lambda v: 4 if v=="Slab" else v)
    testDF.loc[:, 'Foundation'] = testDF.loc[:, 'Foundation'].map(lambda v: 5 if v=="Stone" else v)
    testDF.loc[:, 'Foundation'] = testDF.loc[:, 'Foundation'].fillna(trainDF.loc[:, 'Foundation'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #BsmtQual Values: NaN, 'Gd', 'TA', 'Ex', 'Fa'
    #Preprocessing method: NaN = Mode(trainDf), 'Gd'=0, 'TA'=1, 'Ex'=2, 'Fa'=3
    trainDF.loc[:, 'BsmtQual'] = trainDF.loc[:, 'BsmtQual'].map(lambda v: 3 if v=="Ex" else v)
    trainDF.loc[:, 'BsmtQual'] = trainDF.loc[:, 'BsmtQual'].map(lambda v: 2 if v=="Gd" else v)
    trainDF.loc[:, 'BsmtQual'] = trainDF.loc[:, 'BsmtQual'].map(lambda v: 1 if v=="TA" else v)
    trainDF.loc[:, 'BsmtQual'] = trainDF.loc[:, 'BsmtQual'].map(lambda v: 0 if v=="Fa" else v)
    trainDF.loc[:, 'BsmtQual'] = trainDF.loc[:, 'BsmtQual'].fillna(-1)
    
    testDF.loc[:, 'BsmtQual'] = testDF.loc[:, 'BsmtQual'].map(lambda v: 3 if v=="Ex" else v)
    testDF.loc[:, 'BsmtQual'] = testDF.loc[:, 'BsmtQual'].map(lambda v: 2 if v=="Gd" else v)
    testDF.loc[:, 'BsmtQual'] = testDF.loc[:, 'BsmtQual'].map(lambda v: 1 if v=="TA" else v)
    testDF.loc[:, 'BsmtQual'] = testDF.loc[:, 'BsmtQual'].map(lambda v: 0 if v=="Fa" else v)
    testDF.loc[:, 'BsmtQual'] = testDF.loc[:, 'BsmtQual'].fillna(-1)
    
        #----------------------------------------------------------------------------------------------------
    
    #BsmtCond Values: NaN, 'TA', 'Gd', 'Fa', 'Po'
    #Preprocessing method: NaN = Mode(trainDf), 'TA'=0, 'Gd'=1, 'Fa'=2, 'Po'=3
    trainDF.loc[:, 'BsmtCond'] = trainDF.loc[:, 'BsmtCond'].map(lambda v: 3 if v=="Gd" else v)
    trainDF.loc[:, 'BsmtCond'] = trainDF.loc[:, 'BsmtCond'].map(lambda v: 2 if v=="TA" else v)
    trainDF.loc[:, 'BsmtCond'] = trainDF.loc[:, 'BsmtCond'].map(lambda v: 1 if v=="Fa" else v)
    trainDF.loc[:, 'BsmtCond'] = trainDF.loc[:, 'BsmtCond'].map(lambda v: 0 if v=="Po" else v)
    trainDF.loc[:, 'BsmtCond'] = trainDF.loc[:, 'BsmtCond'].fillna(trainDF.loc[:, 'BsmtCond'].mode().iloc[0])
    
    testDF.loc[:, 'BsmtCond'] = testDF.loc[:, 'BsmtCond'].map(lambda v: 3 if v=="Gd" else v)
    testDF.loc[:, 'BsmtCond'] = testDF.loc[:, 'BsmtCond'].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:, 'BsmtCond'] = testDF.loc[:, 'BsmtCond'].map(lambda v: 1 if v=="Fa" else v)
    testDF.loc[:, 'BsmtCond'] = testDF.loc[:, 'BsmtCond'].map(lambda v: 0 if v=="Po" else v)
    testDF.loc[:, 'BsmtCond'] = testDF.loc[:, 'BsmtCond'].fillna(-1)
    
        #----------------------------------------------------------------------------------------------------
    
    #BsmtExposure Values: NaN, 'No', 'Gd', 'Mn', 'Av'
    #Preprocessing method: NaN = Mode(trainDf), 'No'=0, 'Gd'=1, 'Mn'=2, 'Av'=3
    trainDF.loc[:, 'BsmtExposure'] = trainDF.loc[:, 'BsmtExposure'].map(lambda v: 0 if v=="No" else v)
    trainDF.loc[:, 'BsmtExposure'] = trainDF.loc[:, 'BsmtExposure'].map(lambda v: 1 if v=="Gd" else v)
    trainDF.loc[:, 'BsmtExposure'] = trainDF.loc[:, 'BsmtExposure'].map(lambda v: 2 if v=="Mn" else v)
    trainDF.loc[:, 'BsmtExposure'] = trainDF.loc[:, 'BsmtExposure'].map(lambda v: 3 if v=="Av" else v)
    trainDF.loc[:, 'BsmtExposure'] = trainDF.loc[:, 'BsmtExposure'].fillna(-1)
    
    testDF.loc[:, 'BsmtExposure'] = testDF.loc[:, 'BsmtExposure'].map(lambda v: 0 if v=="No" else v)
    testDF.loc[:, 'BsmtExposure'] = testDF.loc[:, 'BsmtExposure'].map(lambda v: 1 if v=="Gd" else v)
    testDF.loc[:, 'BsmtExposure'] = testDF.loc[:, 'BsmtExposure'].map(lambda v: 2 if v=="Mn" else v)
    testDF.loc[:, 'BsmtExposure'] = testDF.loc[:, 'BsmtExposure'].map(lambda v: 3 if v=="Av" else v)
    testDF.loc[:, 'BsmtExposure'] = testDF.loc[:, 'BsmtExposure'].fillna(-1)
    
        #----------------------------------------------------------------------------------------------------
    
    #BsmtFinType1 Values: NaN, 'GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ'
    #Preprocessing Method: NaN = Mode(trainDf), 'GLQ'=0, 'ALQ'=1, 'Unf'=2, 'Rec'=3, 'BLQ'=4, 'LwQ'=5
    trainDF.loc[:, 'BsmtFinType1'] = trainDF.loc[:, 'BsmtFinType1'].map(lambda v: 0 if v=="GLQ" else v)
    trainDF.loc[:, 'BsmtFinType1'] = trainDF.loc[:, 'BsmtFinType1'].map(lambda v: 1 if v=="ALQ" else v)
    trainDF.loc[:, 'BsmtFinType1'] = trainDF.loc[:, 'BsmtFinType1'].map(lambda v: 2 if v=="Unf" else v)
    trainDF.loc[:, 'BsmtFinType1'] = trainDF.loc[:, 'BsmtFinType1'].map(lambda v: 3 if v=="Rec" else v)
    trainDF.loc[:, 'BsmtFinType1'] = trainDF.loc[:, 'BsmtFinType1'].map(lambda v: 4 if v=="BLQ" else v)
    trainDF.loc[:, 'BsmtFinType1'] = trainDF.loc[:, 'BsmtFinType1'].map(lambda v: 5 if v=="LwQ" else v)
    trainDF.loc[:, 'BsmtFinType1'] = trainDF.loc[:, 'BsmtFinType1'].fillna(-1)
    
    testDF.loc[:, 'BsmtFinType1'] = testDF.loc[:, 'BsmtFinType1'].map(lambda v: 0 if v=="GLQ" else v)
    testDF.loc[:, 'BsmtFinType1'] = testDF.loc[:, 'BsmtFinType1'].map(lambda v: 1 if v=="ALQ" else v)
    testDF.loc[:, 'BsmtFinType1'] = testDF.loc[:, 'BsmtFinType1'].map(lambda v: 2 if v=="Unf" else v)
    testDF.loc[:, 'BsmtFinType1'] = testDF.loc[:, 'BsmtFinType1'].map(lambda v: 3 if v=="Rec" else v)
    testDF.loc[:, 'BsmtFinType1'] = testDF.loc[:, 'BsmtFinType1'].map(lambda v: 4 if v=="BLQ" else v)
    testDF.loc[:, 'BsmtFinType1'] = testDF.loc[:, 'BsmtFinType1'].map(lambda v: 5 if v=="LwQ" else v)
    testDF.loc[:, 'BsmtFinType1'] = testDF.loc[:, 'BsmtFinType1'].fillna(-1)
    
        #----------------------------------------------------------------------------------------------------
    
    #BsmtFinType2 Values: NaN, 'GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ'
    #Preprocessing Method: NaN = Mode(trainDf), 'GLQ'=0, 'ALQ'=1, 'Unf'=2, 'Rec'=3, 'BLQ'=4, 'LwQ'=5
    trainDF.loc[:, 'BsmtFinType2'] = trainDF.loc[:, 'BsmtFinType2'].map(lambda v: 0 if v=="GLQ" else v)
    trainDF.loc[:, 'BsmtFinType2'] = trainDF.loc[:, 'BsmtFinType2'].map(lambda v: 1 if v=="ALQ" else v)
    trainDF.loc[:, 'BsmtFinType2'] = trainDF.loc[:, 'BsmtFinType2'].map(lambda v: 2 if v=="Unf" else v)
    trainDF.loc[:, 'BsmtFinType2'] = trainDF.loc[:, 'BsmtFinType2'].map(lambda v: 3 if v=="Rec" else v)
    trainDF.loc[:, 'BsmtFinType2'] = trainDF.loc[:, 'BsmtFinType2'].map(lambda v: 4 if v=="BLQ" else v)
    trainDF.loc[:, 'BsmtFinType2'] = trainDF.loc[:, 'BsmtFinType2'].map(lambda v: 5 if v=="LwQ" else v)
    trainDF.loc[:, 'BsmtFinType2'] = trainDF.loc[:, 'BsmtFinType2'].fillna(-1)
    
    testDF.loc[:, 'BsmtFinType2'] = testDF.loc[:, 'BsmtFinType2'].map(lambda v: 0 if v=="GLQ" else v)
    testDF.loc[:, 'BsmtFinType2'] = testDF.loc[:, 'BsmtFinType2'].map(lambda v: 1 if v=="ALQ" else v)
    testDF.loc[:, 'BsmtFinType2'] = testDF.loc[:, 'BsmtFinType2'].map(lambda v: 2 if v=="Unf" else v)
    testDF.loc[:, 'BsmtFinType2'] = testDF.loc[:, 'BsmtFinType2'].map(lambda v: 3 if v=="Rec" else v)
    testDF.loc[:, 'BsmtFinType2'] = testDF.loc[:, 'BsmtFinType2'].map(lambda v: 4 if v=="BLQ" else v)
    testDF.loc[:, 'BsmtFinType2'] = testDF.loc[:, 'BsmtFinType2'].map(lambda v: 5 if v=="LwQ" else v)
    testDF.loc[:, 'BsmtFinType2'] = testDF.loc[:, 'BsmtFinType2'].fillna(-1)
    
        #----------------------------------------------------------------------------------------------------
    
    #Electrical Values: NaN, 'SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix'
    #Preprocessing Method: NaN = Mode(trainDf), 'SBrkr'=0, 'FuseF'=1, 'FuseA'=2, 'FuseP'=3, 'Mix'=4
    trainDF.loc[:, 'Electrical'] = trainDF.loc[:, 'Electrical'].map(lambda v: 0 if v=="SBrkr" else v)
    trainDF.loc[:, 'Electrical'] = trainDF.loc[:, 'Electrical'].map(lambda v: 1 if v=="FuseF" else v)
    trainDF.loc[:, 'Electrical'] = trainDF.loc[:, 'Electrical'].map(lambda v: 2 if v=="FuseA" else v)
    trainDF.loc[:, 'Electrical'] = trainDF.loc[:, 'Electrical'].map(lambda v: 3 if v=="FuseP" else v)
    trainDF.loc[:, 'Electrical'] = trainDF.loc[:, 'Electrical'].map(lambda v: 4 if v=="Mix" else v)
    trainDF.loc[:, 'Electrical'] = trainDF.loc[:, 'Electrical'].fillna(trainDF.loc[:, 'Electrical'].mode().iloc[0])
    
    testDF.loc[:, 'Electrical'] = testDF.loc[:, 'Electrical'].map(lambda v: 0 if v=="SBrkr" else v)
    testDF.loc[:, 'Electrical'] = testDF.loc[:, 'Electrical'].map(lambda v: 1 if v=="FuseF" else v)
    testDF.loc[:, 'Electrical'] = testDF.loc[:, 'Electrical'].map(lambda v: 2 if v=="FuseA" else v)
    testDF.loc[:, 'Electrical'] = testDF.loc[:, 'Electrical'].map(lambda v: 3 if v=="FuseP" else v)
    testDF.loc[:, 'Electrical'] = testDF.loc[:, 'Electrical'].map(lambda v: 4 if v=="Mix" else v)
    testDF.loc[:, 'Electrical'] = testDF.loc[:, 'Electrical'].fillna(trainDF.loc[:, 'Electrical'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #KitchenQual Values: NaN, 'Gd', 'TA', 'Ex', 'Fa'
    #Preprocessing Method: NaN = Mode(trainDf), 'Gd'=0, 'TA'=1, 'Ex'=2, 'Fa'=3
    trainDF.loc[:, 'KitchenQual'] = trainDF.loc[:, 'KitchenQual'].map(lambda v: 3 if v=="Ex" else v)
    trainDF.loc[:, 'KitchenQual'] = trainDF.loc[:, 'KitchenQual'].map(lambda v: 2 if v=="Gd" else v)
    trainDF.loc[:, 'KitchenQual'] = trainDF.loc[:, 'KitchenQual'].map(lambda v: 1 if v=="TA" else v)
    trainDF.loc[:, 'KitchenQual'] = trainDF.loc[:, 'KitchenQual'].map(lambda v: 0 if v=="Fa" else v)
    trainDF.loc[:, 'KitchenQual'] = trainDF.loc[:, 'KitchenQual'].fillna(trainDF.loc[:, 'KitchenQual'].mode().iloc[0])
    
    testDF.loc[:, 'KitchenQual'] = testDF.loc[:, 'KitchenQual'].map(lambda v: 3 if v=="Ex" else v)
    testDF.loc[:, 'KitchenQual'] = testDF.loc[:, 'KitchenQual'].map(lambda v: 2 if v=="Gd" else v)
    testDF.loc[:, 'KitchenQual'] = testDF.loc[:, 'KitchenQual'].map(lambda v: 1 if v=="TA" else v)
    testDF.loc[:, 'KitchenQual'] = testDF.loc[:, 'KitchenQual'].map(lambda v: 0 if v=="Fa" else v)
    testDF.loc[:, 'KitchenQual'] = testDF.loc[:, 'KitchenQual'].fillna(trainDF.loc[:, 'KitchenQual'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #FireplaceQu values: NaN, 'TA', 'Gd', 'Fa', 'Po', 'Ex'
    #Preprocessing Method: NaN = Mode(train), Gd = 0, TA = 1, Ex = 2, Fa = 3, Po = 4
    trainDF.loc[:, 'FireplaceQu'] = trainDF.loc[:, 'FireplaceQu'].map(lambda v: 4 if v=="Ex" else v)
    trainDF.loc[:, 'FireplaceQu'] = trainDF.loc[:, 'FireplaceQu'].map(lambda v: 3 if v=="Gd" else v)
    trainDF.loc[:, 'FireplaceQu'] = trainDF.loc[:, 'FireplaceQu'].map(lambda v: 2 if v=="TA" else v)
    trainDF.loc[:, 'FireplaceQu'] = trainDF.loc[:, 'FireplaceQu'].map(lambda v: 1 if v=="Fa" else v)
    trainDF.loc[:, 'FireplaceQu'] = trainDF.loc[:, 'FireplaceQu'].map(lambda v: 0 if v=="Po" else v)
    trainDF.loc[:, 'FireplaceQu'] = trainDF.loc[:, 'FireplaceQu'].fillna(trainDF.loc[:, 'FireplaceQu'].mode().iloc[0])
    
    testDF.loc[:, 'FireplaceQu'] = testDF.loc[:, 'FireplaceQu'].map(lambda v: 4 if v=="Ex" else v)
    testDF.loc[:, 'FireplaceQu'] = testDF.loc[:, 'FireplaceQu'].map(lambda v: 3 if v=="Gd" else v)
    testDF.loc[:, 'FireplaceQu'] = testDF.loc[:, 'FireplaceQu'].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:, 'FireplaceQu'] = testDF.loc[:, 'FireplaceQu'].map(lambda v: 1 if v=="Fa" else v)
    testDF.loc[:, 'FireplaceQu'] = testDF.loc[:, 'FireplaceQu'].map(lambda v: 0 if v=="Po" else v)
    testDF.loc[:, 'FireplaceQu'] = testDF.loc[:, 'FireplaceQu'].fillna(trainDF.loc[:, 'FireplaceQu'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
        #                              GARAGE ATTRIBUTES
        #----------------------------------------------------------------------------------------------------
    
    #GarageType Values: NaN, 'Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', '2Types'
    #Preprocessing Method: NaN = Mode(trainDf), 'Attchd'=0, 'Detchd'=1, 'BuiltIn'=2, 'CarPort'=3, 'Basment'=4, '2Types'=5
    trainDF.loc[:, 'GarageType'] = trainDF.loc[:, 'GarageType'].map(lambda v: 0 if v=="Attchd" else v)
    trainDF.loc[:, 'GarageType'] = trainDF.loc[:, 'GarageType'].map(lambda v: 1 if v=="Detchd" else v)
    trainDF.loc[:, 'GarageType'] = trainDF.loc[:, 'GarageType'].map(lambda v: 2 if v=="BuiltIn" else v)
    trainDF.loc[:, 'GarageType'] = trainDF.loc[:, 'GarageType'].map(lambda v: 3 if v=="CarPort" else v)
    trainDF.loc[:, 'GarageType'] = trainDF.loc[:, 'GarageType'].map(lambda v: 4 if v=="Basment" else v)
    trainDF.loc[:, 'GarageType'] = trainDF.loc[:, 'GarageType'].map(lambda v: 5 if v=="2Types" else v)
    trainDF.loc[:, 'GarageType'] = trainDF.loc[:, 'GarageType'].fillna(-1)
    
    testDF.loc[:, 'GarageType'] = testDF.loc[:, 'GarageType'].map(lambda v: 0 if v=="Attchd" else v)
    testDF.loc[:, 'GarageType'] = testDF.loc[:, 'GarageType'].map(lambda v: 1 if v=="Detchd" else v)
    testDF.loc[:, 'GarageType'] = testDF.loc[:, 'GarageType'].map(lambda v: 2 if v=="BuiltIn" else v)
    testDF.loc[:, 'GarageType'] = testDF.loc[:, 'GarageType'].map(lambda v: 3 if v=="CarPort" else v)
    testDF.loc[:, 'GarageType'] = testDF.loc[:, 'GarageType'].map(lambda v: 4 if v=="Basment" else v)
    testDF.loc[:, 'GarageType'] = testDF.loc[:, 'GarageType'].map(lambda v: 5 if v=="2Types" else v)
    testDF.loc[:, 'GarageType'] = testDF.loc[:, 'GarageType'].fillna(-1)
    
        #----------------------------------------------------------------------------------------------------
    
    #GarageFinish Values: NaN, 'RFn', 'Unf', 'Fin'
    #Preprocessing Method: NaN = Mode(trainDf), 'RFn'=0, 'Unf'=1, 'Fin'=2
    trainDF.loc[:, 'GarageFinish'] = trainDF.loc[:, 'GarageFinish'].map(lambda v: 0 if v=="RFn" else v)
    trainDF.loc[:, 'GarageFinish'] = trainDF.loc[:, 'GarageFinish'].map(lambda v: 1 if v=="Unf" else v)
    trainDF.loc[:, 'GarageFinish'] = trainDF.loc[:, 'GarageFinish'].map(lambda v: 2 if v=="Fin" else v)
    trainDF.loc[:, 'GarageFinish'] = trainDF.loc[:, 'GarageFinish'].fillna(trainDF.loc[:, 'GarageFinish'].mode().iloc[0])
    
    testDF.loc[:, 'GarageFinish'] = testDF.loc[:, 'GarageFinish'].map(lambda v: 0 if v=="RFn" else v)
    testDF.loc[:, 'GarageFinish'] = testDF.loc[:, 'GarageFinish'].map(lambda v: 1 if v=="Unf" else v)
    testDF.loc[:, 'GarageFinish'] = testDF.loc[:, 'GarageFinish'].map(lambda v: 2 if v=="Fin" else v)
    testDF.loc[:, 'GarageFinish'] = testDF.loc[:, 'GarageFinish'].fillna(trainDF.loc[:, 'GarageFinish'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #GarageQual values: NaN, 'TA', 'Gd', 'Fa', 'Po', 'Ex'
    #Preprocessing Method: NaN = Mode(train), Gd = 0, TA = 1, Ex = 2, Fa = 3, Po = 4
    trainDF.loc[:, 'GarageQual'] = trainDF.loc[:, 'GarageQual'].map(lambda v: 4 if v=="Ex" else v)
    trainDF.loc[:, 'GarageQual'] = trainDF.loc[:, 'GarageQual'].map(lambda v: 3 if v=="Gd" else v)
    trainDF.loc[:, 'GarageQual'] = trainDF.loc[:, 'GarageQual'].map(lambda v: 2 if v=="TA" else v)
    trainDF.loc[:, 'GarageQual'] = trainDF.loc[:, 'GarageQual'].map(lambda v: 1 if v=="Fa" else v)
    trainDF.loc[:, 'GarageQual'] = trainDF.loc[:, 'GarageQual'].map(lambda v: 0 if v=="Po" else v)
    trainDF.loc[:, 'GarageQual'] = trainDF.loc[:, 'GarageQual'].fillna(trainDF.loc[:, 'GarageQual'].mode().iloc[0])
    
    testDF.loc[:, 'GarageQual'] = testDF.loc[:, 'GarageQual'].map(lambda v: 4 if v=="Ex" else v)
    testDF.loc[:, 'GarageQual'] = testDF.loc[:, 'GarageQual'].map(lambda v: 3 if v=="Gd" else v)
    testDF.loc[:, 'GarageQual'] = testDF.loc[:, 'GarageQual'].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:, 'GarageQual'] = testDF.loc[:, 'GarageQual'].map(lambda v: 1 if v=="Fa" else v)
    testDF.loc[:, 'GarageQual'] = testDF.loc[:, 'GarageQual'].map(lambda v: 0 if v=="Po" else v)
    testDF.loc[:, 'GarageQual'] = testDF.loc[:, 'GarageQual'].fillna(trainDF.loc[:, 'GarageQual'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #GarageCond values: NaN, 'TA', 'Gd', 'Fa', 'Po', 'Ex'
    #Preprocessing Method: NaN = Mode(train), Gd = 0, TA = 1, Ex = 2, Fa = 3, Po = 4
    trainDF.loc[:, 'GarageCond'] = trainDF.loc[:, 'GarageCond'].map(lambda v: 4 if v=="Ex" else v)
    trainDF.loc[:, 'GarageCond'] = trainDF.loc[:, 'GarageCond'].map(lambda v: 3 if v=="Gd" else v)
    trainDF.loc[:, 'GarageCond'] = trainDF.loc[:, 'GarageCond'].map(lambda v: 2 if v=="TA" else v)
    trainDF.loc[:, 'GarageCond'] = trainDF.loc[:, 'GarageCond'].map(lambda v: 1 if v=="Fa" else v)
    trainDF.loc[:, 'GarageCond'] = trainDF.loc[:, 'GarageCond'].map(lambda v: 0 if v=="Po" else v)
    trainDF.loc[:, 'GarageCond'] = trainDF.loc[:, 'GarageCond'].fillna(trainDF.loc[:, 'GarageCond'].mode().iloc[0])
    
    testDF.loc[:, 'GarageCond'] = testDF.loc[:, 'GarageCond'].map(lambda v: 4 if v=="Ex" else v)
    testDF.loc[:, 'GarageCond'] = testDF.loc[:, 'GarageCond'].map(lambda v: 3 if v=="Gd" else v)
    testDF.loc[:, 'GarageCond'] = testDF.loc[:, 'GarageCond'].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:, 'GarageCond'] = testDF.loc[:, 'GarageCond'].map(lambda v: 1 if v=="Fa" else v)
    testDF.loc[:, 'GarageCond'] = testDF.loc[:, 'GarageCond'].map(lambda v: 0 if v=="Po" else v)
    testDF.loc[:, 'GarageCond'] = testDF.loc[:, 'GarageCond'].fillna(trainDF.loc[:, 'GarageCond'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #PavedDrive values: NaN, 'Y', 'N', 'P'
    #Preprocessing Method: NaN = Mode(train), 'Y'=0, 'N'=1, 'P'=2
    trainDF.loc[:, 'PavedDrive'] = trainDF.loc[:, 'PavedDrive'].map(lambda v: 0 if v=="Y" else v)
    trainDF.loc[:, 'PavedDrive'] = trainDF.loc[:, 'PavedDrive'].map(lambda v: 1 if v=="N" else v)
    trainDF.loc[:, 'PavedDrive'] = trainDF.loc[:, 'PavedDrive'].map(lambda v: 2 if v=="P" else v)
    trainDF.loc[:, 'PavedDrive'] = trainDF.loc[:, 'PavedDrive'].fillna(trainDF.loc[:, 'PavedDrive'].mode().iloc[0])
    
    testDF.loc[:, 'PavedDrive'] = testDF.loc[:, 'PavedDrive'].map(lambda v: 0 if v=="Y" else v)
    testDF.loc[:, 'PavedDrive'] = testDF.loc[:, 'PavedDrive'].map(lambda v: 1 if v=="N" else v)
    testDF.loc[:, 'PavedDrive'] = testDF.loc[:, 'PavedDrive'].map(lambda v: 2 if v=="P" else v)
    testDF.loc[:, 'PavedDrive'] = testDF.loc[:, 'PavedDrive'].fillna(trainDF.loc[:, 'PavedDrive'].mode().iloc[0])
    
        #---------------------------------------------------------------------------------------------------- 
    
    #PoolQC values: NaN, 'Ex', 'Fa', 'Gd'
    #Preprocessing Method: NaN = Mode(train), 'Ex'=0, 'Fa'=1, 'Gd'=2
    trainDF.loc[:, 'PoolQC'] = trainDF.loc[:, 'PoolQC'].map(lambda v: 2 if v=="Ex" else v)
    trainDF.loc[:, 'PoolQC'] = trainDF.loc[:, 'PoolQC'].map(lambda v: 1 if v=="Gd" else v)
    trainDF.loc[:, 'PoolQC'] = trainDF.loc[:, 'PoolQC'].map(lambda v: 0 if v=="Fa" else v)
    trainDF.loc[:, 'PoolQC'] = trainDF.loc[:, 'PoolQC'].fillna(trainDF.loc[:, 'PoolQC'].mode().iloc[0])
    
    testDF.loc[:, 'PoolQC'] = testDF.loc[:, 'PoolQC'].map(lambda v: 2 if v=="Ex" else v)
    testDF.loc[:, 'PoolQC'] = testDF.loc[:, 'PoolQC'].map(lambda v: 1 if v=="Gd" else v)
    testDF.loc[:, 'PoolQC'] = testDF.loc[:, 'PoolQC'].map(lambda v: 0 if v=="Fa" else v)
    testDF.loc[:, 'PoolQC'] = testDF.loc[:, 'PoolQC'].fillna(trainDF.loc[:, 'PoolQC'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #Fence values: NaN, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'
    #Preprocessing Method: NaN = Mode(train), 'MnPrv'=0, 'GdWo'=1, 'GdPrv'=2, 'MnWw'=3
    trainDF.loc[:, 'Fence'] = trainDF.loc[:, 'Fence'].map(lambda v: 0 if v=="MnPrv" else v)
    trainDF.loc[:, 'Fence'] = trainDF.loc[:, 'Fence'].map(lambda v: 1 if v=="GdWo" else v)
    trainDF.loc[:, 'Fence'] = trainDF.loc[:, 'Fence'].map(lambda v: 2 if v=="GdPrv" else v)
    trainDF.loc[:, 'Fence'] = trainDF.loc[:, 'Fence'].map(lambda v: 3 if v=="MnWw" else v)
    trainDF.loc[:, 'Fence'] = trainDF.loc[:, 'Fence'].fillna(trainDF.loc[:, 'Fence'].mode().iloc[0])
    
    testDF.loc[:, 'Fence'] = testDF.loc[:, 'Fence'].map(lambda v: 0 if v=="MnPrv" else v)
    testDF.loc[:, 'Fence'] = testDF.loc[:, 'Fence'].map(lambda v: 1 if v=="GdWo" else v)
    testDF.loc[:, 'Fence'] = testDF.loc[:, 'Fence'].map(lambda v: 2 if v=="GdPrv" else v)
    testDF.loc[:, 'Fence'] = testDF.loc[:, 'Fence'].map(lambda v: 3 if v=="MnWw" else v)
    testDF.loc[:, 'Fence'] = testDF.loc[:, 'Fence'].fillna(trainDF.loc[:, 'Fence'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #MiscFeature values: NaN, 'Shed', 'Gar2', 'Othr', 'TenC'
    #Preprocessing Method: NaN = Mode(train), 'Shed'=0, 'Gar2'=1, 'Othr'=2, 'TenC'=3
    trainDF.loc[:, 'MiscFeature'] = trainDF.loc[:, 'MiscFeature'].map(lambda v: 0 if v=="Shed" else v)
    trainDF.loc[:, 'MiscFeature'] = trainDF.loc[:, 'MiscFeature'].map(lambda v: 1 if v=="Gar2" else v)
    trainDF.loc[:, 'MiscFeature'] = trainDF.loc[:, 'MiscFeature'].map(lambda v: 2 if v=="Othr" else v)
    trainDF.loc[:, 'MiscFeature'] = trainDF.loc[:, 'MiscFeature'].map(lambda v: 3 if v=="TenC" else v)
    trainDF.loc[:, 'MiscFeature'] = trainDF.loc[:, 'MiscFeature'].fillna(trainDF.loc[:, 'MiscFeature'].mode().iloc[0])
    
    testDF.loc[:, 'MiscFeature'] = testDF.loc[:, 'MiscFeature'].map(lambda v: 0 if v=="Shed" else v)
    testDF.loc[:, 'MiscFeature'] = testDF.loc[:, 'MiscFeature'].map(lambda v: 1 if v=="Gar2" else v)
    testDF.loc[:, 'MiscFeature'] = testDF.loc[:, 'MiscFeature'].map(lambda v: 2 if v=="Othr" else v)
    testDF.loc[:, 'MiscFeature'] = testDF.loc[:, 'MiscFeature'].map(lambda v: 3 if v=="TenC" else v)
    testDF.loc[:, 'MiscFeature'] = testDF.loc[:, 'MiscFeature'].fillna(trainDF.loc[:, 'MiscFeature'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #SaleType values: NaN, 'WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth'
    #Preprocessing Method: NaN = Mode(train), 'WD'=0, 'New'=1, 'COD'=2, 'ConLD'=3, 'ConLI'=4, 'CWD'=5, 'ConLw'=6, 
    #                           'Con'=7, 'Oth'=8
    trainDF.loc[:, 'SaleType'] = trainDF.loc[:, 'SaleType'].map(lambda v: 0 if v=="WD" else v)
    trainDF.loc[:, 'SaleType'] = trainDF.loc[:, 'SaleType'].map(lambda v: 1 if v=="New" else v)
    trainDF.loc[:, 'SaleType'] = trainDF.loc[:, 'SaleType'].map(lambda v: 2 if v=="COD" else v)
    trainDF.loc[:, 'SaleType'] = trainDF.loc[:, 'SaleType'].map(lambda v: 3 if v=="ConLD" else v)
    trainDF.loc[:, 'SaleType'] = trainDF.loc[:, 'SaleType'].map(lambda v: 4 if v=="ConLI" else v)
    trainDF.loc[:, 'SaleType'] = trainDF.loc[:, 'SaleType'].map(lambda v: 5 if v=="CWD" else v)
    trainDF.loc[:, 'SaleType'] = trainDF.loc[:, 'SaleType'].map(lambda v: 6 if v=="ConLw" else v)
    trainDF.loc[:, 'SaleType'] = trainDF.loc[:, 'SaleType'].map(lambda v: 7 if v=="Con" else v)
    trainDF.loc[:, 'SaleType'] = trainDF.loc[:, 'SaleType'].map(lambda v: 8 if v=="Oth" else v)
    trainDF.loc[:, 'SaleType'] = trainDF.loc[:, 'SaleType'].fillna(trainDF.loc[:, 'SaleType'].mode().iloc[0])
    
    testDF.loc[:, 'SaleType'] = testDF.loc[:, 'SaleType'].map(lambda v: 0 if v=="WD" else v)
    testDF.loc[:, 'SaleType'] = testDF.loc[:, 'SaleType'].map(lambda v: 1 if v=="New" else v)
    testDF.loc[:, 'SaleType'] = testDF.loc[:, 'SaleType'].map(lambda v: 2 if v=="COD" else v)
    testDF.loc[:, 'SaleType'] = testDF.loc[:, 'SaleType'].map(lambda v: 3 if v=="ConLD" else v)
    testDF.loc[:, 'SaleType'] = testDF.loc[:, 'SaleType'].map(lambda v: 4 if v=="ConLI" else v)
    testDF.loc[:, 'SaleType'] = testDF.loc[:, 'SaleType'].map(lambda v: 5 if v=="CWD" else v)
    testDF.loc[:, 'SaleType'] = testDF.loc[:, 'SaleType'].map(lambda v: 6 if v=="ConLw" else v)
    testDF.loc[:, 'SaleType'] = testDF.loc[:, 'SaleType'].map(lambda v: 7 if v=="Con" else v)
    testDF.loc[:, 'SaleType'] = testDF.loc[:, 'SaleType'].map(lambda v: 8 if v=="Oth" else v)
    testDF.loc[:, 'SaleType'] = testDF.loc[:, 'SaleType'].fillna(trainDF.loc[:, 'SaleType'].mode().iloc[0])
    
        #----------------------------------------------------------------------------------------------------
    
    #SaleCondition values: NaN, 'Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'
    # Preprocessing Method: NaN = Mode(train),  'Normal'=0, 'Abnorml'=1, 'Partial'=2, 'AdjLand'=3, 'Alloca'=4, 'Family=5'
    trainDF.loc[:, 'SaleCondition'] = trainDF.loc[:, 'SaleCondition'].map(lambda v: 0 if v=="Normal" else v)
    trainDF.loc[:, 'SaleCondition'] = trainDF.loc[:, 'SaleCondition'].map(lambda v: 1 if v=="Abnorml" else v)
    trainDF.loc[:, 'SaleCondition'] = trainDF.loc[:, 'SaleCondition'].map(lambda v: 2 if v=="Partial" else v)
    trainDF.loc[:, 'SaleCondition'] = trainDF.loc[:, 'SaleCondition'].map(lambda v: 3 if v=="AdjLand" else v)
    trainDF.loc[:, 'SaleCondition'] = trainDF.loc[:, 'SaleCondition'].map(lambda v: 4 if v=="Alloca" else v)
    trainDF.loc[:, 'SaleCondition'] = trainDF.loc[:, 'SaleCondition'].map(lambda v: 5 if v=="Family" else v)
    trainDF.loc[:, 'SaleCondition'] = trainDF.loc[:, 'SaleCondition'].fillna(trainDF.loc[:, 'SaleCondition'].mode().iloc[0])
    
    testDF.loc[:, 'SaleCondition'] = testDF.loc[:, 'SaleCondition'].map(lambda v: 0 if v=="Normal" else v)
    testDF.loc[:, 'SaleCondition'] = testDF.loc[:, 'SaleCondition'].map(lambda v: 1 if v=="Abnorml" else v)
    testDF.loc[:, 'SaleCondition'] = testDF.loc[:, 'SaleCondition'].map(lambda v: 2 if v=="Partial" else v)
    testDF.loc[:, 'SaleCondition'] = testDF.loc[:, 'SaleCondition'].map(lambda v: 3 if v=="AdjLand" else v)
    testDF.loc[:, 'SaleCondition'] = testDF.loc[:, 'SaleCondition'].map(lambda v: 4 if v=="Alloca" else v)
    testDF.loc[:, 'SaleCondition'] = testDF.loc[:, 'SaleCondition'].map(lambda v: 5 if v=="Family" else v)
    testDF.loc[:, 'SaleCondition'] = testDF.loc[:, 'SaleCondition'].fillna(trainDF.loc[:, 'SaleCondition'].mode().iloc[0])
    
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    
    visualization(trainDF, predictors)
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors

# ===============================================================================
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')
    
    print("Attributes containing NaN:", getNaAttrs(trainDF))
    
    # #These attributes were breaking the model, so we did a test to see which values they contained
    problems = ['BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath','GarageCars', 'GarageArea']
    print("Values, for problem attributes:", getAttrToValuesDictionary(trainDF.loc[:, problems]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================
'''
Modification of the numeric attributes method
Returns the attributes with NA values.
'''

def getNaAttrs(df):
    return __getNumericHelper(df, False)

def __getNaHelper(df, findNan):
    isNan = df.applymap(np.isnan) # np.isnan is a function that takes a value and returns True (the value is na) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is na (numeric) or not

    isNan = isNan.all() # all: For each column, returns whether all elements are True
    attrs = isNan.loc[isNan==findNan].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs


# =============================================================================
'''
Correlation method
Returns the correlation coefficient between each attribute and the SalePrice
'''
def findCorr(df):
    corr = df.corr(method='pearson')
    corr = corr.loc[:,'SalePrice']
    corr = corr.map(lambda v: 'VP' if (v > 0.7 or v < -0.7 )else ("I" if (v > 0.5 or v < -0.5)else ("R" if (v > 0.3 or v < -0.3) else "N")))
    # corr = corr.apply(lambda col: 'VP' if col > 0.7 else ("I" if col > 0.6 else ("R" if col > 0.5 else "N")), axis=0)
    return corr
# =============================================================================
'''
Using Ideas from HW 5 to find the best value for K
'''
def paramSearchPlot(inputDF, outputSeries):
    neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
    
    accuracies = neighborList.map(lambda row: model_selection.cross_val_score(KNeighborsRegressor(n_neighbors = row), inputDF, outputSeries, cv=10, scoring='r2').mean())
    print(accuracies)

    plt.plot(neighborList, accuracies)
    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    
    print(neighborList.loc[accuracies.idxmax()])
# =============================================================================
'''
Get a model of the correlation between the attributes
'''
# def visualization(trainDF, inputCols):    
#     sns.set()
#     cols = ['OverallQual', 'GrLivArea', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'ExterQual', 'BsmtQual', 'KitchenQual']
#     sns.pairplot(trainDF[cols], height = 2.5)
#     plt.show()
    
def visualization(trainDF, inputCols):    
    sns.set()
    cols = ['OverallQual', 'GrLivArea', 'SalePrice']
    sns.pairplot(trainDF[cols], height = 2.5)
    plt.show()


if __name__ == "__main__":
    main()

