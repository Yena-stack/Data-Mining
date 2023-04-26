import pandas as pd
import numpy as np
import scipy as sp
import scipy.spatial
import timeit
import math

'''
Use the default value of numRows (None) to read *all* rows
'''
def readData(numRows = None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows = numRows)
    return wineDF, inputCols, outputCol

def euclideanDist1(s1, s2):
    sum = 0
    for i in range(s1.size):
        sum += (s1.iloc[i]-s2.iloc[i])**2
    return math.sqrt(sum)
        
def euclideanDist2(p1, p2):
    newSeries = pd.Series(range(p1.size))
    return math.sqrt(newSeries.map(lambda x: (p1.iloc[x]-p2.iloc[x])**2).sum())
                  
def euclideanDist3(s1, s2):
    _sum = (s1-s2)**2
    return math.sqrt(_sum.sum())
    
def euclideanDist4(p1, p2):
    return np.linalg.norm(p1-p2)

def euclideanDist5(p1, p2):
    return scipy.spatial.distance.euclidean(p1, p2)

def main():
    
    #NUMBER 7
    #EUCLIDIAN DISTANCE METHODS, SLOWEST TO FASTEST FOR 1000 LOOP:
    # 2
    # 3
    # 1
    # 4
    # 5
    
    #NUMBER 8
    #EUCLIDIAN DISTANCE METHODS, SLOWEST TO FASTEST FOR 1000 LOOP WITH EXTRA COLS:
    # 2
    # 1
    # 3
    # 4
    # 5
    
    #NUMBER 9
    
    
    df, inputCols, outputCol = readData(3)

    def addRandomCols(df, numNewCols):
        newCols = pd.Series(['rndC'+str(i) for i in range(0, numNewCols)]) 
        newCols.map(lambda colName: addRandomCol(colName, df))

    def addRandomCol(colName, df):
        df.loc[:, colName] = np.random.randint(-100, 100, df.shape[0])
    
    addRandomCols(df, 100)

    a = df.iloc[0, :]
    b = df.iloc[1, :]
    c = df.iloc[2, :]
    
    test04()
    print("============================== 1")
    startTime = timeit.default_timer()
    for i in range(1000):
        euclideanDist1(a,b)
        euclideanDist1(a,c)
        euclideanDist1(b,c)
    # print(euclideanDist1(a, b))
    # print(euclideanDist1(a, c))
    # print(euclideanDist1(b, c))  
    elapsedTime = timeit.default_timer() - startTime
    print(elapsedTime)
    
    print("============================== 2")
    startTime = timeit.default_timer()
    for i in range(1000):
        euclideanDist2(a,b)
        euclideanDist2(a,c)
        euclideanDist2(b,c)
    # print(euclideanDist2(a, b))
    # print(euclideanDist2(a, c))
    # print(euclideanDist2(b, c))  
    elapsedTime = timeit.default_timer() - startTime
    print(elapsedTime)
    
    print("============================== 3")
    startTime = timeit.default_timer()
    for i in range(1000):
        euclideanDist3(a,b)
        euclideanDist3(a,c)
        euclideanDist3(b,c)    
    # print(euclideanDist3(a, b))
    # print(euclideanDist3(a, c))
    # print(euclideanDist3(b, c))  
    elapsedTime = timeit.default_timer() - startTime
    print(elapsedTime)

    print("============================== 4")    
    startTime = timeit.default_timer()
    for i in range(1000):
        euclideanDist4(a,b)
        euclideanDist4(a,c)
        euclideanDist4(b,c)
    # print(euclideanDist4(a, b))
    # print(euclideanDist4(a, c))
    # print(euclideanDist4(b, c))  
    elapsedTime = timeit.default_timer() - startTime
    print(elapsedTime)

    print("============================== 5")    
    startTime = timeit.default_timer()
    for i in range(1000):
        euclideanDist5(a,b)
        euclideanDist5(a,c)
        euclideanDist5(b,c)    
    # print(euclideanDist5(a, b))
    # print(euclideanDist5(a, c))
    # print(euclideanDist5(b, c))  
    elapsedTime = timeit.default_timer() - startTime
    print(elapsedTime)
    


def test04():
    df, inputCols, outputCol = readData(3)
    a = df.iloc[0, :]
    c = df.iloc[2, :]

    print(euclideanDist1(a, c))
    print(euclideanDist2(a, c))
    print(euclideanDist3(a, c))
    print(euclideanDist4(a, c))
    print(euclideanDist5(a, c))
