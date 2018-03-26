import dataLoad
import drawTools 
import matplotlib.pyplot as plt
import naiveBayes
import discretization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import crossValidation
import numpy as np
import pandas as pd


def countStatistics(_y_true, _y_pred):
    try:
        accracy = accuracy_score(_y_true, _y_pred, normalize=True)
        precision = precision_score(_y_true, _y_pred, average='binary')
        recall = recall_score(_y_true, _y_pred, average='binary')
        fscore = f1_score(_y_true, _y_pred, average='binary')
    except:
        accracy = accuracy_score(_y_true, _y_pred, normalize=True)
        precision = precision_score(_y_true, _y_pred, average='weighted')
        recall = recall_score(_y_true, _y_pred, average='weighted')
        fscore = f1_score(_y_true, _y_pred, average='weighted')
    
    return accracy, precision, recall, fscore

def fullAnalise(_df):

    classes = [
        'Multinomial no discretization',
        'Multinomial equal width',
        'Multinomial equal freq',
        'MDLP',
        'Gausian'
    ]

    features = [
        'accuracy',
        'precision',
        'recall',
        'fscore',
        'class'
    ]

    statsMap = {}
    for x in features:
        statsMap[x] = []
    
    mClass = 'Multinomial no discretization'
    print(mClass)
    allTargets, allPreds = crossValidation.normal(_df, 9, 'multinomial', None, True, True)
    accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
    statsMap['accuracy'].append(accuracy)
    statsMap['precision'].append(precision)
    statsMap['recall'].append(recall)
    statsMap['fscore'].append(fscore)
    statsMap['class'].append(mClass)
    print(countStatistics(allTargets, allPreds))
    drawTools.drawConfusionMatrix(allTargets, allPreds, pltNumb=1, title='No discretization')

    mClass = 'Multinomial equal width'
    print(mClass)
    allTargets, allPreds = crossValidation.normal(_df, 9, 'multinomial', 'width', True, True)
    accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
    statsMap['accuracy'].append(accuracy)
    statsMap['precision'].append(precision)
    statsMap['recall'].append(recall)
    statsMap['fscore'].append(fscore)
    statsMap['class'].append(mClass)
    print(countStatistics(allTargets, allPreds))
    drawTools.drawConfusionMatrix(allTargets, allPreds, pltNumb=2, title='Equal width')

    mClass = 'Multinomial equal freq'
    print(mClass)
    allTargets, allPreds = crossValidation.normal(_df, 9, 'multinomial', 'freq', True, True)
    accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
    statsMap['accuracy'].append(accuracy)
    statsMap['precision'].append(precision)
    statsMap['recall'].append(recall)
    statsMap['fscore'].append(fscore)
    statsMap['class'].append(mClass)
    print(countStatistics(allTargets, allPreds))   
    drawTools.drawConfusionMatrix(allTargets, allPreds, pltNumb=3, title='Equal Frequency')

    mClass = 'MDLP'
    print(mClass)
    allTargets, allPreds = crossValidation.normal(_df, 9, 'multinomial', 'mdpl', True, True)
    accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
    statsMap['accuracy'].append(accuracy)
    statsMap['precision'].append(precision)
    statsMap['recall'].append(recall)
    statsMap['fscore'].append(fscore)
    statsMap['class'].append(mClass)
    print(countStatistics(allTargets, allPreds))   
    drawTools.drawConfusionMatrix(allTargets, allPreds, pltNumb=4, title='MDPL')

    mClass = 'Gausian'
    print(mClass)
    allTargets, allPreds = crossValidation.normal(_df, 9, 'gausian', None, True, True)
    accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
    statsMap['accuracy'].append(accuracy)
    statsMap['precision'].append(precision)
    statsMap['recall'].append(recall)
    statsMap['fscore'].append(fscore)
    statsMap['class'].append(mClass)
    print(countStatistics(allTargets, allPreds))
    drawTools.drawConfusionMatrix(allTargets, allPreds, pltNumb=5, title='Gausian')

    mDf = pd.DataFrame.from_dict(statsMap)
    drawTools.drawBars(mDf)


def drawDiscretization(_df):
    drawTools.drawHistDiagonal(discretization.equalWidth(_df))
    drawTools.drawHistDiagonal(discretization.equalFreq(_df))
    drawTools.drawHistDiagonal(discretization.discMdlp(_df))

def testFoldX(_df, _shuffle, _stratified):
    x = 2
    allTargets, allPreds = crossValidation.normal(_df, x, 'gausian', None, _shuffle, _stratified)
    print(countStatistics(allTargets, allPreds))

    x = 3
    allTargets, allPreds = crossValidation.normal(_df, x, 'gausian', None, _shuffle, _stratified)
    print(countStatistics(allTargets, allPreds))

    x = 4
    allTargets, allPreds = crossValidation.normal(_df, x, 'gausian', None, _shuffle, _stratified)
    print(countStatistics(allTargets, allPreds))

    x = 5
    allTargets, allPreds = crossValidation.normal(_df, x, 'gausian', None, _shuffle, _stratified)
    print(countStatistics(allTargets, allPreds))

    x = 6
    allTargets, allPreds = crossValidation.normal(_df, x, 'gausian', None, _shuffle, _stratified)
    print(countStatistics(allTargets, allPreds))

    x = 7
    allTargets, allPreds = crossValidation.normal(_df, x, 'gausian', None, _shuffle, _stratified)
    print(countStatistics(allTargets, allPreds))


    x = 8
    allTargets, allPreds = crossValidation.normal(_df, x, 'gausian', None, _shuffle, _stratified)
    print(countStatistics(allTargets, allPreds))


    x = 9
    allTargets, allPreds = crossValidation.normal(_df, x, 'gausian', None, _shuffle, _stratified)
    print(countStatistics(allTargets, allPreds))

    x = 10
    allTargets, allPreds = crossValidation.normal(_df, x, 'gausian', None, _shuffle, _stratified)
    print(countStatistics(allTargets, allPreds))
    

dataSet = dataLoad.getDataSetWine()
fullAnalise(dataSet)

# print('@@Wine')
# print('##Brak randomizacji, brak stratyfikacji')
# testFoldX(dataSet, False, False)
# print('##Randomizacja, brak stratyfikacji')
# testFoldX(dataSet, True, False)
# print('##Brak randomizacji, stratyfikacja')
# testFoldX(dataSet, False, True)
# print('##Randomizacja, stratyfikacja')
# testFoldX(dataSet, True, True)

# dataSet = dataLoad.getDataSetGlass()
# print('@@Glass')
# print('##Brak randomizacji, brak stratyfikacji')
# testFoldX(dataSet, False, False)
# print('##Randomizacja, brak stratyfikacji')
# testFoldX(dataSet, True, False)
# print('##Brak randomizacji, stratyfikacja')
# testFoldX(dataSet, False, True)
# print('##Randomizacja, stratyfikacja')
# testFoldX(dataSet, True, True)

# dataSet = dataLoad.getDataSetDiabetes()
# print('@@Diabetes')
# print('##Brak randomizacji, brak stratyfikacji')
# testFoldX(dataSet, False, False)
# print('##Randomizacja, brak stratyfikacji')
# testFoldX(dataSet, True, False)
# print('##Brak randomizacji, stratyfikacja')
# testFoldX(dataSet, False, True)
# print('##Randomizacja, stratyfikacja')
# testFoldX(dataSet, True, True)

# data load testing

# dataSet = dataLoad.getDataSetDiabetes()
# drawTools.drawHistDiagonal(dataSet)
# plt.show()

# clsifier testing

# dataSet = dataLoad.getDataSetIris()
# classifier = naiveBayes.getClassifireGausian(dataSet)
# yPred = naiveBayes.predict(classifier, dataSet)
# print(yPred)
# print("Number of mislabeled points out of a total %d points : %d" % (len(dataSet),(dataSet['Class'] != yPred).sum()))




# disc testing
# dataSet = dataLoad.getDataSetWine()
# drawDiscretization(dataSet)

# dataSet = dataLoad.getDataSetDiabetes()
# drawDiscretization(dataSet)
# analiseDiscretization(dataSet)
# discretization.discMdlp(dataSet)
# print('countStatistics(allTargets, allPreds)')
# print(countStatistics(allTargets, allPreds))

# print("Number of mislabeled points out of a total %d points : %d" % (len(allTargets),(np.array(allTargets) != np.array(allPreds)).sum()))
plt.show()
