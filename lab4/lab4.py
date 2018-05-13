import dataLoad
import drawTools 
import matplotlib.pyplot as plt
import naiveBayes
import discretization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import crossValidation
import numpy as np
import pandas as pd



# Metrics
EUCLIDEAN = 'euclidean'
MANHATAN = 'manhattan'

# voting methods
UNIFORM = 'uniform'
DISTANCE = 'distance'
SQUARED_DISTANCE = 'squaredDistances'

def countStatistics(_y_true, _y_pred):
    try:
        accracy = round(accuracy_score(_y_true, _y_pred, normalize=True),3)
        precision = round(precision_score(_y_true, _y_pred, average='binary'),3)
        recall = round(recall_score(_y_true, _y_pred, average='binary'),3)
        fscore = round(f1_score(_y_true, _y_pred, average='binary'),3)
    except:
        accracy = round(accuracy_score(_y_true, _y_pred, normalize=True),3)
        precision = round(precision_score(_y_true, _y_pred, average='weighted'),3)
        recall = round(recall_score(_y_true, _y_pred, average='weighted'),3)
        fscore = round(f1_score(_y_true, _y_pred, average='weighted'),3)
    
    return accracy, precision, recall, fscore


mDataSet = dataLoad.getDataSetGlass()
allTargets, allPreds = crossValidation.normal(mDataSet, 5, 5, MANHATAN, DISTANCE)
accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
print(fscore)


# for x in [2,3,4,5,7,10,15,20,50]:
#     allTargets, allPreds = crossValidation.normal(mDataSet, 5, x, MANHATAN, UNIFORM)
#     accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
#     print ("{k} & {accuracy} & {precision} & {recall} & {fscore}\\\\".format(k=x, accuracy=accuracy, precision=precision, recall=recall, fscore=fscore))


# print('DS Knowledge')
# print('Voting ' + UNIFORM)
# allTargets, allPreds = crossValidation.normal(mDataSet, 5, 5, MANHATAN, UNIFORM)
# print(countStatistics(allTargets, allPreds))
# drawTools.drawConfusionMatrix(allTargets, allPreds, pltNumb=1, title='Voting ' + UNIFORM)

# print('Voting ' + DISTANCE)
# allTargets, allPreds = crossValidation.normal(mDataSet, 5, 5, MANHATAN, DISTANCE)
# print(countStatistics(allTargets, allPreds))
# drawTools.drawConfusionMatrix(allTargets, allPreds, pltNumb=2, title='Voting ' + DISTANCE)

# print('Voting ' + SQUARED_DISTANCE)
# allTargets, allPreds = crossValidation.normal(mDataSet, 5, 5, MANHATAN, SQUARED_DISTANCE)
# print(countStatistics(allTargets, allPreds))
# drawTools.drawConfusionMatrix(allTargets, allPreds, pltNumb=3, title='Voting ' + SQUARED_DISTANCE)

plt.show()