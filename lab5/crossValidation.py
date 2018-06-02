from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import copy

def crossVal(_df, _parts, _classifier):

    mData = _df

    featureVals = [x for x in mData if x != 'Class']
    kf = StratifiedKFold(n_splits=_parts, shuffle=False)

    allTargets = []
    allPreds = []

    for train_index, test_index in kf.split(mData, mData['Class']):
        tarinSet = mData.iloc[train_index]

        examSet = mData.iloc[test_index]
        examTargets = mData.iloc[test_index]['Class']

        mClassifier = copy.deepcopy(_classifier)
        
        mClassifier.fit(tarinSet[featureVals], tarinSet['Class'])
        predTargets = mClassifier.predict(examSet[featureVals])

        allTargets.extend(examTargets)
        allPreds.extend(predTargets)

    return allTargets, allPreds
        
        
