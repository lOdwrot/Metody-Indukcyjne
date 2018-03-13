import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
from discretization import discretizationEqualWidth, discretizationQuantileTransform, discretizationEqualFreq
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from math import factorial, sqrt, ceil
from sklearn.metrics import confusion_matrix
import dataLoad

dNone = 'None'
dEqualWitdh = 'Equal Width'
dEqualFreq = 'Equal Freq'
dQuantile = 'Quantile Transform'

def drawDataSet(_data, _targets, _paramsDesc, _fig, _sign, _markerSize, _alpha=1):
    plt.figure(_fig.number)
    plotMatrix = (np.arange(len(_data[0])**2) + 1).reshape(len(_data[0]), len(_data[0]))
    subplotNumb = 1
    plotsInRow = ceil(sqrt(
        factorial(len(_data[0])) / (factorial(len(_data[0]) -2)*2)
        ))
    

    # calculate plots numbers
    for i in [x for x in range(0, len(_data[0]))]:
        for j in [x for x in range(i + 1, len(_data[0]))]:
            plotMatrix[i][j] = subplotNumb
            subplotNumb = subplotNumb + 1

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

    # describe axis
    if _paramsDesc is not None:
        for i in [x for x in range(0, len(_data[0] - 2))]:
                for j in [x for x in range(i + 1, len(_data[0] - 1))]:
                    ax = plt.subplot(plotsInRow, plotsInRow, plotMatrix[i][j])
                    ax.set_xlabel(_paramsDesc[i])
                    ax.set_ylabel(_paramsDesc[j])

    # draw points
    for sampleNumb in range(0, len(_data)):
        for i in [x for x in range(0, len(_data[0] ))]:
            for j in [x for x in range(i + 1, len(_data[0]))]:
                ax = plt.subplot(plotsInRow, plotsInRow, plotMatrix[i][j])
                ax.plot(_data[sampleNumb][i], _data[sampleNumb][j], colors[_targets[sampleNumb]] + _sign, markersize =_markerSize, alpha=_alpha)

def getClassifireNormal(_features, _targets):
    gnb = GaussianNB()
    gnb.fit(_features, _targets)
    return gnb

def getClassifireSmooth(_features, _targets):
    gnbM = MultinomialNB(alpha=1.0)
    gnbM.fit(_features, _targets)
    return gnbM

def getFeatureByTypes(_features, _targets, _shuffle=False):
    result = {}
    for i in np.unique(_targets):
        result[i] = []
    for feature, target in zip(_features, _targets):
        result[target].append(feature)
    
    if _shuffle is True:
        for key, val in result.items():
            random.shuffle(val)

    return result

def foldX(_featuresMap, _parts):
    result = {}
    for key, vals in _featuresMap.items():
        partsTab = []
        partLen = int(len(vals) / _parts)
        for i in range(0, _parts - 1):
            partsTab.append(vals[i*partLen : (i+1)*partLen])
        partsTab.append(vals[(_parts - 1) * partLen : ])
        result[key] = partsTab
    return result

def foldXarr(_featuresArray, _foldTargets, _parts):
    featueTab = []
    targetTab = []

    partLen = int(len(_featuresArray) / _parts)
    for i in range(0, _parts - 1):
        featueTab.append(_featuresArray[i*partLen : (i+1)*partLen])
        targetTab.append(_foldTargets[i*partLen : (i+1)*partLen])

    return featueTab, targetTab



# def crossValidation(_features, _targets, _classifierType, _discretizationTechnique, _groups):
#     print('Cross validation lunched')
#     allExamTargets = []
#     allExamResults = []

#     # discretization
#     if _discretizationTechnique == dEqualWitdh:
#         _features = discretizationEqualWidth(_features, 10)
#     elif _discretizationTechnique == dEqualFreq:
#         _features = discretizationEqualFreq(_features)
#     elif _discretizationTechnique == dQuantile:
#         _features = discretizationQuantileTransform(_features)

#     byTypes = getFeatureByTypes(_features, _targets, True)
#     splitted = foldX(byTypes, _groups)
    

#     for examSubsetNumb in range(0, len(splitted[0])):
#         learnSet = []
#         learnTargets = []
#         examSet = []
#         examTargets = []

#         for key in splitted:
#             parts = splitted[key]
#             for j in range(0, len(parts)):
#                 if examSubsetNumb != j:
#                     learnSet.extend(parts[j])
#                     learnTargets.extend([key for i in range(0, len(parts[j]))])
#                 else:
#                     examSet.extend(parts[j])
#                     examTargets.extend([key for i in range(0, len(parts[j]))])
#         if _classifierType == 'normal':
#             classifier = getClassifireNormal(learnSet, learnTargets)
#         else:
#             classifier = getClassifireSmooth(learnSet, learnTargets)

#         y_pred = classifier.predict(examSet)
#         allExamTargets.extend(examTargets)
#         allExamResults.extend(y_pred)

#     return allExamTargets, allExamResults

def crossValidationNS(_features, _targets, _classifierType, _discretizationTechnique, _groups):
    print('Cross validation lunched')
    allExamTargets = []
    allExamResults = []

    # discretization
    if _discretizationTechnique == dEqualWitdh:
        _features = discretizationEqualWidth(_features, 10)
    elif _discretizationTechnique == dEqualFreq:
        _features = discretizationEqualFreq(_features)
    elif _discretizationTechnique == dQuantile:
        _features = discretizationQuantileTransform(_features)
 
    splitted, splittedTarget = foldXarr(_features, _targets, _groups)
    parts = splitted
    
    

    for examSubsetNumb in range(0, int(len(_features) / _groups) - 1):
        learnSet = []
        learnTargets = []
        examSet = []
        examTargets = []

        for j in range(0, len(parts) - 1):
            if examSubsetNumb != j:
                learnSet.extend(parts[j])
                learnTargets.extend(splittedTarget[j])
            else:
                examSet.extend(parts[j])
                examTargets.extend(splittedTarget[j])
        if _classifierType == 'normal':

            classifier = getClassifireNormal(learnSet, learnTargets)
        else:
            classifier = getClassifireSmooth(learnSet, learnTargets)
 
        print(examSet)
        print(examTargets)

        if len(examSet) > 0:
            y_pred = classifier.predict(examSet)

            allExamTargets.extend(examTargets)
            allExamResults.extend(y_pred)

    return allExamTargets, allExamResults



def drawConfusionMatrix(y_True, y_Pred, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, pltNumb=None):
    if pltNumb is None:
        fig = plt.figure()
    else:
        plt.subplot(2,2,pltNumb)

    cm = confusion_matrix(y_True, y_Pred)
    if classes is None:    
        if min(y_True) < min(y_Pred): 
            mMin = min(y_True)
        else:
            mMin = min(y_Pred) 

        if max(y_True) > max(y_Pred): 
            mMax = max(y_True)
        else:
            mMax = max(y_Pred) 
        classes = list(range(mMin, mMax + 1))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def countStatistics(_y_true, _y_pred, _draw=True):
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

def drawMergedStats(_paramNames, _barVals, _legendLabels=None, tiltle=None):
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r']
    alphas = [1, 1, 1, 1, 1, 1, 1, 0.5]
    baseWidth = 0.2
    mMax = max([max(x) for x in _barVals]) + 0.05
    if mMax > 1:
        mMax = 1
    ax.set_ylim(min([min(x) for x in _barVals]) - 0.05, mMax)

    indexes = list(range(0, len(_paramNames)))
    for index, val in enumerate(_barVals):
        nIndexes = [x + index*baseWidth for x in indexes]
        if _legendLabels is None:
            ax.bar(nIndexes, val, color=colors[index%(len(colors)-1)], alpha=alphas[index%(len(colors)-1)], width=baseWidth)
        else:
            ax.bar(nIndexes, val, color=colors[index%(len(colors)-1)], alpha=alphas[index%(len(colors)-1)], width=baseWidth, label=_legendLabels[index])
            plt.legend()
    
    if tiltle is not None:
        plt.title(tiltle)
    plt.xticks([x + 1.75 * baseWidth for x in indexes], _paramNames)
    


def analise(_features, _targets, _classifireType, _setName, _featureNames, _discretizationTechnique, _pltNumb=None, _foldParts=10):
    # reults without discretization
    print("Start analised " + _setName)

    y_true, y_pred = crossValidationNS(_features, _targets, _classifireType, _discretizationTechnique, _foldParts)
    drawConfusionMatrix(y_true, y_pred, classes=_featureNames, normalize="True", title=_discretizationTechnique, pltNumb=_pltNumb)
    accracy, precision, recall, fscore = countStatistics(y_true, y_pred)
    print("accracy: %f, precision: %f , recall: %f, fscore: %f" % (accracy, precision, recall, fscore))
    
    print("End analised " + _setName)
    return [accracy, precision, recall, fscore]

def discretizationComparation(_features, _targets, _classifireType, _setName, _featureNames):
    mStats = []
    mLabels = ['None', 'Equal Width', 'Equal Freq', 'Quantile Transform']
    mStats.append(analise(_features, _targets, _classifireType, _setName,_featureNames, dNone, 1))
    mStats.append(analise(_features, _targets, _classifireType, _setName,_featureNames, dEqualWitdh, 2))
    mStats.append(analise(_features, _targets, _classifireType, _setName,_featureNames, dEqualFreq, 3))
    mStats.append(analise(_features, _targets, _classifireType, _setName,_featureNames, dQuantile, 4))
    drawMergedStats(['accuracy', 'precision', 'recall', 'fscore'], mStats, mLabels, _setName)

def foldComparation(_features, _targets, _classifireType, _setName, _featureNames):
    mStats = []
    mLabels = ['Fold 2', 'Fold 3', 'Fold 5', 'Fold 10']
    mStats.append(analise(_features, _targets, _classifireType, _setName,_featureNames, mLabels[0], 1, 2))
    # mStats.append(analise(_features, _targets, _classifireType, _setName,_featureNames, mLabels[1], 2, 3))
    # mStats.append(analise(_features, _targets, _classifireType, _setName,_featureNames, mLabels[2], 3, 5))
    # mStats.append(analise(_features, _targets, _classifireType, _setName,_featureNames, mLabels[3], 4, 10))
    
    # drawMergedStats(['accuracy', 'precision', 'recall', 'fscore'], mStats, mLabels, _setName)

def debug(): 
    figIris = plt.figure()
    dataSet = datasets.load_iris()
    drawDataSet(dataSet['data'], dataSet['target'], dataSet['feature_names'], figIris, 'o', 5, 1)
    discretizationComparation(dataSet['data'], dataSet['target'], "normal", "Wine", dataSet["target_names"])
    foldComparation(dataSet['data'], dataSet['target'], "multinomial", "Iris", dataSet["target_names"])

    
    


#end 
dataSetName = "Iris"
dataSet = datasets.load_iris()
# dataSetName = "Wine"
# dataSet = datasets.load_wine()
# dataSetName = "Diabetes"
# dataSet = dataLoad.getDiabetesDataSet()

# discretizationComparation(dataSet['data'], dataSet['target'], "normal", dataSetName, dataSet["target_names"])
foldComparation(dataSet['data'], dataSet['target'], "normal", dataSetName, dataSet["target_names"])

# draw data Set
# mFig = plt.figure()
# drawDataSet(discretizationEqualWidth(dataSet['data'], 10), dataSet['target'], dataSet['feature_names'], mFig, 'o', 5, 1)

# mFig = plt.figure()
# drawDataSet(discretizationEqualFreq(dataSet['data']), dataSet['target'], dataSet['feature_names'], mFig, 'o', 5, 1)

# mFig = plt.figure()
# drawDataSet(discretizationQuantileTransform(dataSet['data']), dataSet['target'], dataSet['feature_names'], mFig, 'o', 5, 1)

plt.show()

