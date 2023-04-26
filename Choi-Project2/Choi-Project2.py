# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import math


def getAgeList(numRows = None):
    colNames = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    origData = pd.read_csv("data/adult.data", delimiter = ", ", header=None, names=colNames, engine='python', nrows=numRows)
    return origData.loc[:, 'age'].tolist()

def calcMean(listOfValues = [0,0]):
    total = 0
    for i in range(0, len(listOfValues)):
        total += listOfValues[i]
        i = i + 1
    return total/len(listOfValues)

def calcStdDev(list = [0,0], mean = 0):
    sum = 0
    for i in list:
        num = (i-mean)*(i-mean)
        sum += num

    return math.sqrt(sum/(len(list)-1))

def calcMode(list):
    d = {}
    for i in list:
        count = 0
        for j in list:
            if(i == j):
                count+=1
        d[i] = count
    mode = 0
    for x in d:
        if d.get(x)>mode:
            mode = x
    
    return mode

def standardize(list):
    mean = calcMean(list)
    stdDev = calcStdDev(list, mean)
    
    #listNew = []
    z = 0
    index = 0
    for e in list:
        z = (e - mean)/stdDev
        list[index] = z
        index += 1
    return list


    
def main():
    ls = [1,4,5,7,4,4,3,2,3]
    #return calcMean(ls)
    #dictionaryPractice()
    #return calcMode(ls)
    #return calcStdDev(ls, calcMean(ls))
    return standardize(ls)


    
