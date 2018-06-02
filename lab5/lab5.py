import dataLoad
from crossValidation import crossVal
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
bagging = BaggingClassifier(GaussianNB(),
                                max_samples=0.5, max_features=0.5)

allTargets, allPreds = crossVal(mDataSet, 5, bagging)
print(countStatistics(allTargets, allPreds))
# featureVals = [x for x in mDataSet if x != 'Class']
# print(bagging.fit(mDataSet[featureVals], mDataSet['Class']).predict(mDataSet[featureVals]))