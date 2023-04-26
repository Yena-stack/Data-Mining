from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy as sp
import pandas as pd
import timeit
import math
import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------------------
# GIVEN: For use in all testing for the purpose of grading
def testMain():
    '''
    This function runs all the tests we'll use for grading. Please don't change it!
    When certain parts need to be graded, uncomment those parts only.
    Please keep all the other parts commented out for grading.
    '''
    pass

    # print("========== testAlwaysOneClassifier ==========")
    # testAlwaysOneClassifier()

    # print("========== testFindNearest() ==========")
    # testFindNearest()

    # print("========== testOneNNClassifier() ==========")
    # testOneNNClassifier()

    # print("========== testCVManual(OneNNClassifier(), 5) ==========")
    # testCVManual(OneNNClassifier(), 5)

    # print("========== testCVBuiltIn(OneNNClassifier(), 5) ==========")
    # testCVBuiltIn(OneNNClassifier(), 5)

    # print("========== compareFolds() ==========")
    # compareFolds()

    # print("========== testStandardize() ==========")
    # testStandardize()

    # print("========== testNormalize() ==========")
    # testNormalize()

    # print("========== comparePreprocessing() ==========")
    # comparePreprocessing()

    # print("========== visualization() ==========")
    # visualization()

    # print("========== testKNN() ==========")
    # testKNN()

    # print("========== paramSearchPlot() ==========")
    # paramSearchPlot()

    # print("========== paramSearchPlotBuiltIn() ==========")
    # paramSearchPlotBuiltIn()
    
    print("========== testSubsets() ==========")
    testSubsets()

# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Reading in the data" step
def readData(numRows=None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids",
                 "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows=numRows)

    # Need to mix this up before doing CV
    wineDF = wineDF.sample(frac=1, random_state=50).reset_index(drop=True)

    return wineDF, inputCols, outputCol
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Testing AlwaysOneClassifier" step
def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
    compare = (actualOutputSeries == predOutputSeries).value_counts()
    # if there are no Trues in compare, then compare[True] throws an error. So we have to check:
    if (True in compare):
        accuracy = compare[True] / actualOutputSeries.size
    else:
        accuracy = 0

    return accuracy
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Standardization on a DataFrame" step
def operationsOnDataFrames():
    d = {'x': pd.Series([1, 2], index=['a', 'b']),
         'y': pd.Series([10, 11], index=['a', 'b']),
         'z': pd.Series([30, 25], index=['a', 'b'])}
    df = pd.DataFrame(d)
    print("Original df:", df, type(df), sep='\n', end='\n\n')

    cols = ['x', 'z']

    df.loc[:, cols] = df.loc[:, cols] / 2
    print("Certain columns / 2:", df, type(df), sep='\n', end='\n\n')

    maxResults = df.loc[:, cols].max()
    print("Max results:", maxResults, type(maxResults), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
def standardize(df, colNames):
    #np.mean(col) 
    #np.std(col)
    df.loc[:, colNames] = df.loc[:, colNames].apply(lambda col: (col-col.mean())/col.std(), axis=0)
    return df

# (row - np.mean(col)/np.std(col))
# GIVEN: For use starting in the "Standardization on a DataFrame" step
def testStandardize():
    df, inputCols, outputCol = readData()
    colsToStandardize = inputCols[2:5]
    print("Before standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    standardize(df, colsToStandardize)
    print("After standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')

    # Proof of standardization:
    print("Means are approx 0:", df.loc[:, colsToStandardize].mean(), sep='\n', end='\n\n')
    print("Stds are approx 1:", df.loc[:, colsToStandardize].std(), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------

def testFindNearest():
    df, inputCols, outputCol = readData()
    startTime = timeit.default_timer()
    for i in range(100):
        findNearestLoop(df.iloc[100:107, :], df.iloc[90, :])
    elapsedTime = timeit.default_timer() - startTime
    print("Loop Time: ")
    print(elapsedTime)
    print("\n")
    print(findNearestLoop(df.iloc[100:107, :], df.iloc[90, :]))
    startTime = timeit.default_timer()
    for i in range(100):
        findNearestHOF(df.iloc[100:107, :], df.iloc[90, :])
    elapsedTime = timeit.default_timer() - startTime
    print("HOF Time: ")
    print(elapsedTime)
    print( findNearestHOF(df.iloc[100:107, :], df.iloc[90, :]))


# -----------------------------------------------------------------------------------------
def normalize(df, colNames):
    df.loc[:, colNames] = df.loc[:, colNames].apply(lambda col: (col-col.min())/(col.max()-col.min()), axis=0)
    return df
    
# GIVEN: For use starting in the "Normalization on a DataFrame" step
def testNormalize():
    df, inputCols, outputCol = readData()
    colsToStandardize = inputCols[2:5]
    print("Before normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    normalize(df, colsToStandardize)
    print("After normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')

    # Proof of normalization:
    print("Maxes are 1:", df.loc[:, colsToStandardize].max(), sep='\n', end='\n\n')
    print("Mins are 0:", df.loc[:, colsToStandardize].min(), sep='\n', end='\n\n')
    
# -----------------------------------------------------------------------------------------
def comparePreprocessing():
    df, inputCols, outputCol = readData()
    inputDF = df.loc[:, inputCols]
    outputSeries = df.loc[:, outputCol]
    scorer = make_scorer(accuracyOfActualVsPredicted,  greater_is_better=True)
    
    inputDFN = normalize(inputDF.copy(), inputCols)
    inputDFS = standardize(inputDF.copy(), inputCols)
    
    accuracies = model_selection.cross_val_score(OneNNClassifier(), inputDF, outputSeries, cv=10, scoring=scorer)
    accuraciesN = model_selection.cross_val_score(OneNNClassifier(), inputDFN, outputSeries, cv=10, scoring=scorer)
    accuraciesS = model_selection.cross_val_score(OneNNClassifier(), inputDFS, outputSeries, cv=10, scoring=scorer)

    print("Average:", np.mean(accuracies))
    print("Average:", np.mean(accuraciesN))
    print("Average:", np.mean(accuraciesS))
    
    '''
    a)
    Average: 0.7477124183006536
    Average: 0.949673202614379
    Average: 0.9552287581699346
    
    The original dataset was the least accurate one since we have not edited the dataset.
    The standardized dataset was the most accurate one 
        since the values are the distance between itself and its mean, which means it is relative,  (better at handles the outlier )
        whereas normalized dataset is just between zero and one, oversimplifying the dataset. 
        if we have same measure, and if we have the same magnitude, it should have higher accuracies. 
    
    b)
    z-transformed data is standardizd data
    
    c)
    leave-one-out technique is essentially the same as the cross validation 
    except it's taking the number of folds that is equal to the number of rows.
            In other words, it's repeating for number of rows times, which makes it the most accurate. 
    higher number of folds , we have bigger training set, which makes it more accurate. 
    '''

# -----------------------------------------------------------------------------------------
def testAlwaysOneClassifier():
    df,inputCols, outputCol = readData()
    testInputDF = df.iloc[:10, 1:]
    testOutputSeries = df.loc[:9, outputCol]
    trainInputDF = df.iloc[10:, 1:]
    trainOutputSeries = df.loc[10:, outputCol]
        
    print("testInputDF:", testInputDF, sep='\n', end='\n\n') 
    print("testOutputSeries:", testOutputSeries, sep='\n', end='\n\n')
    print("trainInputDF:", trainInputDF, sep='\n', end='\n\n') 
    print("trainOutputSeries:", trainOutputSeries, sep='\n', end='\n\n')
    
    test = AlwaysOneClassifier()
    test.fit(trainInputDF, trainOutputSeries)
    
    print("---------- Test one example")
    print("Correct answer: " )
    print (testOutputSeries.iloc[0])
    print("Predicted answer: " )
    print (test.predict(testInputDF.iloc[0, :]), sep='\n', end='\n\n')
    
    print("---------- Test the entire test set")
    print("Correct answer: ")
    print (testOutputSeries, sep='\n', end='\n\n')
    print("Predicted answer: ")
    print (test.predict(testInputDF), sep='\n', end='\n\n')
    
    print ("Accuracy: ")
    print (accuracyOfActualVsPredicted(testOutputSeries.loc[:], (test.predict(testInputDF)).loc[:]))
# -----------------------------------------------------------------------------------------    
def testOneNNClassifier():
    df,inputCols, outputCol = readData()
    testInputDF = df.iloc[:10, 1:]
    testOutputSeries = df.loc[:9, outputCol]
    trainInputDF = df.iloc[10:, 1:]
    trainOutputSeries = df.loc[10:, outputCol]
        
    print("testInputDF:", testInputDF, sep='\n', end='\n\n') 
    print("testOutputSeries:", testOutputSeries, sep='\n', end='\n\n')
    print("trainInputDF:", trainInputDF, sep='\n', end='\n\n') 
    print("trainOutputSeries:", trainOutputSeries, sep='\n', end='\n\n')
    
    test = OneNNClassifier()
    test.fit(trainInputDF, trainOutputSeries)
    
    print("---------- Test one example")
    print("Correct answer: " )
    print (testOutputSeries.iloc[0])
    print("Predicted answer: " )
    print (test.predict(testInputDF.iloc[0, :]), sep='\n', end='\n\n')
    
    print("---------- Test the entire test set")
    print("Correct answer: ")
    print (testOutputSeries, sep='\n', end='\n\n')
    print("Predicted answer: ")
    print (test.predict(testInputDF), sep='\n', end='\n\n')
    
    print ("Accuracy: ")
    print (accuracyOfActualVsPredicted(testOutputSeries, (test.predict(testInputDF))))
    
def testCVManual(model, k):
    # inputDF, inputCols, outputCol = readData()
    df, inputCols, outputCol = readData()
    inputDF = df.loc[:, inputCols]
    outputSeries = df.loc[:, outputCol]
    accuracies = cross_val_score_manual(model, inputDF, outputSeries, k, True)
    
    print("Accuracies:", accuracies) 
    print("Average:", np.mean(accuracies))


def testCVBuiltIn(model, k):
    df, inputCols, outputCol = readData()
    inputDF = df.loc[:, inputCols]
    outputSeries = df.loc[:, outputCol]
    scorer = make_scorer(accuracyOfActualVsPredicted,  greater_is_better=True)
    accuracies = model_selection.cross_val_score(model, inputDF, outputSeries, cv=k, scoring=scorer)
    
    print("Accuracies:", accuracies) 
    print("Average:", np.mean(accuracies))
    
def findNearestLoop(df, testRow):
    small = abs(sp.spatial.distance.euclidean(testRow, df.iloc[0,:]))
    smallIndex = 0
    for i in range(1, df.shape[0]):
        eucDist = abs(sp.spatial.distance.euclidean(testRow, df.iloc[i,:]))
        
        if eucDist<small:
            small = eucDist
            smallIndex = i
        
    return df.index[smallIndex]

def findNearestHOF(df, testRow):
    s = (df.apply(lambda row: sp.spatial.distance.euclidean(testRow, row), axis = 1)).idxmin()
    return s
    
class AlwaysOneClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, inputDF, outputSeries):
        return self
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return 1
        else:
            return pd.Series(np.ones(testInput.shape[0]), index=testInput.index, dtype="int64")
        

class OneNNClassifier(BaseEstimator, ClassifierMixin):
     def __init__(self):
         self.inputDF = None
         self.outputSeries = None
         
     def fit(self, inputDF, outputSeries):
         self.inputDF = inputDF
         self.outputSeries = outputSeries
         return self
     
     def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return self.__predictOne(testInput)
        else:
            return testInput.apply(lambda row: self.__predictOne(row), axis = 1)
     
     def __predictOne(self, testInput):
        label = findNearestHOF(self.inputDF, testInput)
        return self.outputSeries.loc[label]

def cross_val_score_manual(model, inputDF, outputSeries, k, verbose):
    numberOfElements = inputDF.shape[0]
    foldSize = numberOfElements / k
    results = []
    
    for i in range(k):
        testInputDF = None
        testOutputSeries = None
        trainInputDF = None
        trainOutputSeries = None

        start = int(i*foldSize)
        upToNotIncluding = int((i+1)*foldSize)
        
        testInputDF = inputDF.iloc[start:upToNotIncluding, :]
        testOutputSeries = outputSeries.iloc[start:upToNotIncluding]
        
        trainInputDF = pd.concat([inputDF.iloc[:start, :], inputDF.iloc[upToNotIncluding:, :]])
        trainOutputSeries = pd.concat([outputSeries.iloc[:start], outputSeries.iloc[upToNotIncluding:]])
        
        if (verbose):
            print("================================") 
            print("Iteration:", i)
            print("Train input:\n", list(trainInputDF.index))
            print("Train output:\n", list(trainOutputSeries.index)) 
            print("Test input:\n", testInputDF.index)
            print("Test output:\n", testOutputSeries.index)
            print("================================") 
            
        model.fit(trainInputDF, trainOutputSeries)
        predictedOutput = model.predict(testInputDF)

        results.append(accuracyOfActualVsPredicted(testOutputSeries, predictedOutput)) 
    return results

def compareFolds():
    df, inputCols, outputCol = readData()
    inputDF = df.loc[:, inputCols]
    outputSeries = df.loc[:, outputCol]
    
    scorer = make_scorer(accuracyOfActualVsPredicted,  greater_is_better=True)
    accuracies = model_selection.cross_val_score(OneNNClassifier(), inputDF, outputSeries, cv=3, scoring=scorer)
    
    print("Mean accuracy for k=3:", np.mean(accuracies))
    
    scorer = make_scorer(accuracyOfActualVsPredicted,  greater_is_better=True)
    accuracies = model_selection.cross_val_score(OneNNClassifier(), inputDF, outputSeries, cv=10, scoring=scorer)
    
    print("Mean accuracy for k=10:", np.mean(accuracies))

def visualization():
    fullDF, inputCols, outputCol = readData()
    standardize(fullDF, inputCols)
    # sns.displot(fullDF.loc[:, 'Malic Acid'])
    # print(fullDF.loc[:, 'Malic Acid'].skew())
    # sns.displot(fullDF.loc[:, 'Alcohol'])
    # print(fullDF.loc[:, 'Alcohol'].skew())
    # sns.jointplot(x='Malic Acid', y='Alcohol', data=fullDF.loc[:, ['Malic Acid', 'Alcohol']], kind='kde')
    # sns.jointplot(x='Ash', y='Magnesium', data=fullDF.loc[:, ['Ash', 'Magnesium']], kind='kde')
    sns.pairplot(fullDF, hue=outputCol)
    # sns.pairplot(fullDF.loc[:, ['Proline', 'Diluted']], hue=outputCol)


    '''
    a)
    Skew measure: -0.051482331077132064
    It is negatively skewed. 
    
    b)
    The most likely combination of values is (0, -0.5)
    
    c)
    For positive value of Proline, Class 1 was most common.
    
    d)
    If we only use Diluted and Proline, the accuracy will somewhat drop.
    Looking at the pairplot, we can see that Class 1 is located on the upper right and Class 3 is located 
        on the lower left in both graphs (where proline and diluted are used as x-axis and y-axis), so it was able to 
        somewhat classify them using only two attributes. 
    
    e)
    If we only use Nonflavanoid Phenols and Ash, the accuracy will drop significantly.
    Looking at the pairpot, we can see that dots are clustered in the middle for all three classes 
        without a clear pattern in both graphs (where Nonflavanoid Phenols and Ash are used as x-axis and y-axis).
     
    
    f)
    For part (d), the accuracy was 0.8426553672316385 which was around 10-percent drop. 
    For part (e), the accuracy was 0.5447269303201506, and it was a significant drop as we expected.
    '''
    plt.show()
    
def testSubsets():
    fullDF, inputCols, outputCol = readData()
    inputDF = fullDF.loc[:, ['Nonflavanoid Phenols', 'Ash']]
    outputSeries = fullDF.loc[:, outputCol]
    standardize(inputDF, ['Nonflavanoid Phenols', 'Ash'])
    # test = OneNNClassifier()
    scorer = make_scorer(accuracyOfActualVsPredicted,  greater_is_better=True)
    accuracies = model_selection.cross_val_score(OneNNClassifier(), inputDF, outputSeries, cv=3, scoring=scorer)
    print(accuracies.mean())

def testKNN():
    df, inputCols, outputCol = readData()
    inputDF = df.loc[:, inputCols]
    outputSeries = df.loc[:, outputCol]
    
    inputDFS = standardize(inputDF.copy(), inputCols)
    scorer = make_scorer(accuracyOfActualVsPredicted,  greater_is_better=True)
    
    
    accuracies = model_selection.cross_val_score(kNNClassifier(), inputDF, outputSeries, cv=10, scoring=scorer)
    accuraciesS = model_selection.cross_val_score(kNNClassifier(), inputDFS, outputSeries, cv=10, scoring=scorer)
    accuracies8 = model_selection.cross_val_score(kNNClassifier(8), inputDFS, outputSeries, cv=10, scoring=scorer)
    
    print("Unaltered dataset, 1NN, accuracy: ", np.mean(accuracies))
    print("Standardized dataset, 1NN, accuracy: ", np.mean(accuraciesS))
    print("Standardized dataset, 8NN, accuracy: ", np.mean(accuracies8))
    
    
    '''
    8-NN has 8 folds and 1-NN has one fold. Higher number of folds will give higher accuracies because we have a bigger training set. 
    
    '''   
class kNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=1):
         self.k = k
         self.inputDF = None
         self.outputSeries = None
    def fit(self, inputDF, outputSeries):
         self.inputDF = inputDF
         self.outputSeries = outputSeries
         return self
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return self.__predOfKNearest(testInput)
        else:
            return testInput.apply(lambda row: self.__predOfKNearest(row), axis = 1)
    def __predOfKNearest(self, testInput):
        s = (self.inputDF.apply(lambda row: sp.spatial.distance.euclidean(testInput, row), axis = 1)).nsmallest(self.k)
        return self.outputSeries.loc[s.index].mode().iloc[0]
    
def paramSearchPlot():
    neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
    
    df, inputCols, outputCol = readData()
    df = standardize(df, inputCols)
    inputDF = df.loc[:, inputCols]
    outputSeries = df.loc[:, outputCol]
    
    scorer = make_scorer(accuracyOfActualVsPredicted,  greater_is_better=True)
    accuracies = neighborList.map(lambda row: model_selection.cross_val_score(kNNClassifier(row), inputDF, outputSeries, cv=10, scoring=scorer).mean())
    print(accuracies)

    plt.plot(neighborList, accuracies)
    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    
    print(neighborList.loc[accuracies.idxmax()])
    
def paramSearchPlotBuiltIn():
    df, inputCols, outputCol = readData()
    df = standardize(df, inputCols)
    inputDF = df.loc[:, inputCols]
    outputSeries = df.loc[:, outputCol]
    
    alg = KNeighborsClassifier(n_neighbors = 8)
    cvScores = model_selection.cross_val_score(alg, inputDF, outputSeries, cv=10, scoring='accuracy')
    print("Standardized dataset, 8NN, accuracy: ", cvScores.mean())
testMain()