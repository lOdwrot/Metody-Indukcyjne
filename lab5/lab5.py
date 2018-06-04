import dataLoad
from crossValidation import crossVal
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# params
# Bagging

bg_n_estimators = [5,10,15,20,25]
bg_max_samples = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bg_max_features = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Boosting

boost_n_estimators = [5, 10, 20, 30, 40, 50, 60, 70]
boost_learning_rate = [0.8, 0.85, 0.9, 0.95, 1.0]
boost_algorithm = ['SAMME', 'SAMME.R']

# Random Forest

rf_n_estimators = [5,10,15,20,25]
rf_max_features = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rf_max_depth = [3, 4, 5, 6, 7, 8]


def countStatistics(_y_true, _y_pred):
    try:
        accuracy = round(accuracy_score(_y_true, _y_pred, normalize=True),3)
        precision = round(precision_score(_y_true, _y_pred, average='binary'),3)
        recall = round(recall_score(_y_true, _y_pred, average='binary'),3)
        fscore = round(f1_score(_y_true, _y_pred, average='binary'),3)
    except:
        accuracy = round(accuracy_score(_y_true, _y_pred, normalize=True),3)
        precision = round(precision_score(_y_true, _y_pred, average='weighted'),3)
        recall = round(recall_score(_y_true, _y_pred, average='weighted'),3)
        fscore = round(f1_score(_y_true, _y_pred, average='weighted'),3)
    
    return accuracy, precision, recall, fscore

# printing helpers
def drawHeader(featureName):
    print('\hline')
    print(featureName.replace('_', '\_') + ' & Accuracy & Precision & Recall & FScore \\\\')
    print('\hline')

def drawDataSetName(name):
    print('\hline')
    print('\multicolumn{5}{|c|}{' + name + '}\\\\')

def drawRow(val, accuracy, precision, recall, fscore, zeros='2'):
    if not isinstance(val, str):
        print("%(val).2f & %(accuracy).3f & %(precision).3f & %(recall).3f & %(fscore).3f\\\\" % {"val": val, "accuracy":accuracy, "precision":precision, "recall":recall, "fscore":fscore})
    else:
        print(val + " & %(accuracy).3f & %(precision).3f & %(recall).3f & %(fscore).3f\\\\" % {"accuracy":accuracy, "precision":precision, "recall":recall, "fscore":fscore})


mDataSet = dataLoad.getDataSetGlass()
dataSetsMap = {
    'wine': dataLoad.getDataSetWine(),
    'glass': dataLoad.getDataSetGlass(),
    'diabetes': dataLoad.getDataSetDiabetes(),
}

bagging = BaggingClassifier(GaussianNB(),
                                max_samples=0.5, max_features=0.5)

# allTargets, allPreds = crossVal(mDataSet, 5, bagging)
# print(countStatistics(allTargets, allPreds))
# featureVals = [x for x in mDataSet if x != 'Class']
# print(bagging.fit(mDataSet[featureVals], mDataSet['Class']).predict(mDataSet[featureVals]))

# bagging

def printBagging():
    
    for setName in dataSetsMap:
        print('\\begin{tabular}{ |p{3cm}||p{2cm}|p{2cm}|p{2cm}|p{2cm}| }')
        mDataSet = dataSetsMap[setName]
        drawDataSetName(setName)

        drawHeader('n_estimators')
        for v in bg_n_estimators:
            bagging = BaggingClassifier(GaussianNB(), n_estimators = v)
            allTargets, allPreds = crossVal(mDataSet, 5, bagging)
            accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
            drawRow(v, accuracy, precision, recall, fscore)

        drawHeader('max_samples')
        for v in bg_max_samples:
            bagging = BaggingClassifier(GaussianNB(), max_samples = v)
            allTargets, allPreds = crossVal(mDataSet, 5, bagging)
            accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
            drawRow(v, accuracy, precision, recall, fscore)

        drawHeader('max_features')
        for v in bg_max_features:
            bagging = BaggingClassifier(GaussianNB(), max_features = v)
            allTargets, allPreds = crossVal(mDataSet, 5, bagging)
            accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
            drawRow(v, accuracy, precision, recall, fscore)

        print('\hline')
        print('\end{tabular}')
        print('\\\\')
        print()

# boost_n_estimators = [5, 10, 20, 30, 40, 50, 60, 70]
# boost_learning_rate = [0.8, 0.85, 0.9, 0.95, 1.0]
# boost_algorithm = ['SAMME', 'SAMME.R']

def printBoosting():
    
    for setName in dataSetsMap:
        print('\\begin{tabular}{ |p{3cm}||p{2cm}|p{2cm}|p{2cm}|p{2cm}| }')
        mDataSet = dataSetsMap[setName]
        drawDataSetName(setName)

        drawHeader('n_estimators')
        for v in boost_n_estimators:
            mBoost = AdaBoostClassifier(GaussianNB(), n_estimators = v)
            allTargets, allPreds = crossVal(mDataSet, 5, mBoost)
            accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
            drawRow(v, accuracy, precision, recall, fscore)

        drawHeader('learning_rate')
        for v in boost_learning_rate:
            mBoost = AdaBoostClassifier(GaussianNB(), learning_rate = v)
            allTargets, allPreds = crossVal(mDataSet, 5, mBoost)
            accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
            drawRow(v, accuracy, precision, recall, fscore)

        drawHeader('algorithm')
        for v in boost_algorithm:
            mBoost = AdaBoostClassifier(GaussianNB(), algorithm = v)
            allTargets, allPreds = crossVal(mDataSet, 5, mBoost)
            accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
            drawRow(v, accuracy, precision, recall, fscore)

        print('\hline')
        print('\end{tabular}')
        print('\\\\')
        print()    


rf_n_estimators = [5,10,15,20,25]
rf_max_features = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rf_max_depth = [3, 4, 5, 6, 7, 8]

def printRandomForest():
    
    for setName in dataSetsMap:
        print('\\begin{tabular}{ |p{3cm}||p{2cm}|p{2cm}|p{2cm}|p{2cm}| }')
        mDataSet = dataSetsMap[setName]
        drawDataSetName(setName)

        drawHeader('n_estimators')
        for v in rf_n_estimators:
            mRF = RandomForestClassifier(n_estimators = v)
            allTargets, allPreds = crossVal(mDataSet, 5, mRF)
            accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
            drawRow(v, accuracy, precision, recall, fscore)

        drawHeader('max_features')
        for v in rf_max_features:
            mRF = RandomForestClassifier(max_features = v)
            allTargets, allPreds = crossVal(mDataSet, 5, mRF)
            accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
            drawRow(v, accuracy, precision, recall, fscore)

        drawHeader('max_depth')
        for v in rf_max_depth:
            mRF = RandomForestClassifier(max_depth = v)
            allTargets, allPreds = crossVal(mDataSet, 5, mRF)
            accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
            drawRow(v, accuracy, precision, recall, fscore)

        print('\hline')
        print('\end{tabular}')
        print('\\\\')
        print()   


def printBest():
    baggingStats = {
        'wine': [15, 0.8, 0.8],
        'glass': [20, 1.0, 1.0],
        'diabetes': [15, 0.6, 0.8]
    }
    boostingStats = {
        'wine': [70, 0.8, 'SAMME'],
        'glass': [40, 1.0, 'SAMME.R'],
        'diabetes': [10, 0.8, 'SAMME']
    }
    rfStats = {
        'wine': [10, 0.5, 8],
        'glass': [25, 0.6, 8],
        'diabetes': [15, 0.9, 5]
    }

    for setName in dataSetsMap:
        print('######### nDataSet #########')
        print(setName)
        # bagging
        stats = baggingStats[setName]
        mDataSet = dataSetsMap[setName]

        bg = BaggingClassifier(n_estimators = stats[0], max_samples = stats[1], max_features = stats[2])
        allTargets, allPreds = crossVal(mDataSet, 5, bg)
        accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
        print('bagging:')
        print(fscore)

        # boosting
        stats = boostingStats[setName]
        bst = AdaBoostClassifier(n_estimators = stats[0], learning_rate = stats[1], algorithm = stats[2])
        allTargets, allPreds = crossVal(mDataSet, 5, bst)
        accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
        print('boosting')
        print(fscore)

        # random forest
        stats = rfStats[setName]
        rf = RandomForestClassifier(n_estimators = stats[0], max_features = stats[1], max_depth = stats[2])
        allTargets, allPreds = crossVal(mDataSet, 5, rf)
        accuracy, precision, recall, fscore = countStatistics(allTargets, allPreds)
        print('random forest')
        print(fscore)


# printBagging()
# printBoosting()
# printRandomForest()
printBest()

