'''

https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease

Thyroid Disease Data Set

sick.names

sick.data

'''



import pandas as pd

import numpy as np





def readSickData(numRows = None):

    colNames = ["age", 

                "sex", 

                "on thyroxine", 

                "query on thyroxine", 

                "on antithyroid medication",

                "sick", 

                "pregnant", 

                "thyroid surgery",

                "I131 treatment", 

                "query hypothyroid", 

                "query hyperthyroid", 

                "lithium", 

                "goitre", 

                "tumor", 

                "hypopituitary", 

                "psych", 

                "TSH measured", 

                "TSH", 

                "T3 measured", 

                "T3", 

                "TT4 measured", 

                "TT4", 

                "T4U measured", 

                "T4U", 

                "FTI measured", 

                "FTI", 

                "TBG measured", 

                "TBG", 

                "referral source",

                "output"]

    df = pd.read_csv("sick.data", index_col=False, na_values="?", delimiter = ",", header=None, names=colNames, engine='python', nrows=numRows)

    df = df.loc[:, 'age':'referral source']

    return df



def sectionA(first):

    print("\n=================================================================")

    print("=================================================================")

    print("=================================================================")

    print("Section A: Series Access")

    

    print("\nA1 -----------------------------------------------------------------")

    valForColAtPosition2 = first.iloc[17]

    print("Val for col at position 17:", valForColAtPosition2)



    print("\nA2 -----------------------------------------------------------------")

    valForFTI = first.loc["FTI"]

    print("Val for FTI:", valForFTI)

    

    print("\nA3 -----------------------------------------------------------------")

    nameOfCol13 = first.index[13]

    print("Name of col at position 13:", nameOfCol13)

    

    print("\nA4 -----------------------------------------------------------------")

    begToPsych = first.loc[:"psych"]

    print("Beginning to 'psych':\n", begToPsych, sep='')

    

    print("\nA5 -----------------------------------------------------------------")

    col1Through5 = first.iloc[1:6]

    print("Col 1 up to and including 5:\n", col1Through5, sep='')

    

    print("\nA6 -----------------------------------------------------------------")

    col2And5 = first.iloc[[2,5]]

    print("Col 2 and 5:\n", col2And5, sep='')

    

    print("\nA7 -----------------------------------------------------------------")

    tshAndT3 = first.loc[["TSH", "T3"]]

    print("TSH and T3:\n", tshAndT3, sep='')

    

def sectionB(sickDF):

    print("\n=================================================================")

    print("=================================================================")

    print("=================================================================")

    print("Section B: DataFrame Access")

    

    df = readSickData()

    

    print("\nB1 -----------------------------------------------------------------")

    secondPatient = df.iloc[1]

    print("Second patient:\n", secondPatient, sep='')

    

    print("\nB2 -----------------------------------------------------------------")

    preg = df.loc[:,"pregnant"]

    print("Pregnant:\n", preg, sep='')

    

    print("\nB3 -----------------------------------------------------------------")

    allCol24 = df.iloc[[2,4],:]

    print("Patients 2 and 4, all cols:\n", allCol24, sep='')

    

    print("\nB4 -----------------------------------------------------------------")

    someCol24 = df.loc[[2,4], "TSH measured":"TBG"]

    print("Patients 2 and 4, some cols:\n", someCol24, sep="")



def sectionC(sickDF):

    print("\n=================================================================")

    print("=================================================================")

    print("=================================================================")

    print("Section C: DataFrame Processing")

    

    df = readSickData()

    print("\nC1 -----------------------------------------------------------------")

    avgAge = df.loc[:, "age"].mean()

    print("Average age:", avgAge)

    

    print("\nC2 -----------------------------------------------------------------")

    commonSex = df.loc[:, "sex"].mode().iloc[0]

    print("Common sex:", commonSex)

    

    print("\nC3 -----------------------------------------------------------------")

    oldest = df.loc[:, "age"].max()

    print("Oldest age:", oldest)

    

    print("\nC4 -----------------------------------------------------------------")

    oldest3Ages = df.loc[:, "age"].nlargest(3)

    print("Oldest 3 IDs and ages:\n", oldest3Ages, sep="")

    

    print("\nC5 -----------------------------------------------------------------")

    oldest3 = df.iloc[oldest3Ages.index, :]

    print("Oldest 3 all data:\n", oldest3, sep="")

    

    print("\nC6 -----------------------------------------------------------------")

    ageUnder20 = df.loc[df.loc[:,"age"]<20]

    print("Patients under age 20:\n", ageUnder20, sep="")

    

def hw03():

    pd.set_option('display.max_columns', 20)

    

    sickDF = readSickData()

    print("The first 5 rows of the dataset:\n", sickDF.head(5), sep='')

    

    first = sickDF.loc[0, :]

    print("Series object corresponding to the first patient:\n", first, sep='')

    

    sectionA(first)

    sectionB(sickDF)

    sectionC(sickDF)

    

    print("\n=================================================================")

    print("=================================================================")

    print("=================================================================")

    print("Section D: Deeper DataFrame Processing")



    print("\nD1 -----------------------------------------------------------------")

    sickDF = sickDF.loc[sickDF.loc[:,"age"]<120]

    oldest = sickDF.loc[:, "age"].max()

    print("Oldest age after dropping erroneous row:", oldest)

    

    print("\nD2 -----------------------------------------------------------------")

    avgFemaleAge = (sickDF.loc[sickDF.loc[:, "sex"] == "F"].loc[:, "age"]).mean()

    print("Average female age:", avgFemaleAge)

    

    print("\nD3 -----------------------------------------------------------------")

    stdTT4FemaleOver50 = ((sickDF.loc[sickDF.loc[:, "sex"] == "F"].loc[sickDF.loc[:, "age"]>=50]).loc[:, "TT4"]).std()

    print("Standard deviation of TT4 scores for women 50 and older:", stdTT4FemaleOver50)

    

    #HELP

    print("\nD4 -----------------------------------------------------------------")



    bothBoolS = sickDF.loc[((sickDF.loc[:,"on antithyroid medication"]=='t') | (sickDF.loc[:,"thyroid surgery"]=='t')), "sex"]

    countAntiOrSurg = bothBoolS.count()

    print("Number of people on antithyroid med, or with thyroid surgery:", countAntiOrSurg)

    

    print("\nD5 -----------------------------------------------------------------")

    tsh = sickDF.loc[:, "TSH"]

    t3 = sickDF.loc[:, "T3"]

    tt4 = sickDF.loc[:, "TT4"]

    t4u = sickDF.loc[:, "T4U"]

    fti = sickDF.loc[:, "FTI"]

    tbg = sickDF.loc[:, "TBG"]

    

    sickDF.loc[:, 'measurement sum'] = tsh + t3 + tt4 + t4u + fti + tbg

    print("First 5 rows of sickDF after 'measurement sum' column added:\n", sickDF.head(5), sep='')

    

    print("\nD6 -----------------------------------------------------------------")

    sickDF.loc[:, "FTI"] = sickDF.apply(lambda row: max(row.loc["TT4"], 110) if np.isnan(row.loc["FTI"]) else row.loc["FTI"], axis=1)

    print("First 5 rows of sickDF after FTI missing values filled in:\n", sickDF.head(5), sep='')



    print("\nD7 -----------------------------------------------------------------")
    
    sickDF.rename(columns = {"TT4" : "TT4Cat"}, inplace = True)


    sickDF.loc[:, 'TT4Cat'] = sickDF.apply(lambda row: 'low' if row.loc['TT4Cat']<100 else ('high' if row.loc['TT4Cat']>120 else 'medium'), axis = 1) 

    print("First 5 rows of sickDF after replacing TT4 with TT4Cat:\n", sickDF.head(5), sep='')



    print("\nD8 -----------------------------------------------------------------")

    

    tfFemale = sickDF.loc[:, 'sex'] == 'F'

    tf30s = (sickDF.loc[:, 'age'] >= 30) & (sickDF.loc[:, 'age'] < 40)

    tfPreg = sickDF.loc[:, 'pregnant'] == 't'

    tfThyroid = sickDF.loc[:, 'thyroid surgery'] == 'f'

    tfHypo = sickDF.loc[:, 'query hypothyroid'] == 't'

    tfAll = (tfFemale & tf30s) & (tfPreg & tfThyroid) 

    tfAll = tfAll & tfHypo

    

    newSeries = sickDF.loc[tfAll, "sex"]

    areThereAny = newSeries.count() > 0

    print("Are there any such women:", areThereAny)

    

    whoAreThey = newSeries.index

    print("The following women meet those criteria:\n", whoAreThey)

    

    

    print("\nD9 -----------------------------------------------------------------")

    sickDF.loc[:, "on thyroxine"] = sickDF.apply(lambda row: False if row.loc["on thyroxine"] =='f' else True, axis=1)

    print("First 5 rows of sickDF after converting 'on thyroxine' to bool:\n", sickDF.head(5), sep='')

    



    

    

hw03()



def main():

    df = readSickData()

    first = df.iloc[0]

    sectionA(first)