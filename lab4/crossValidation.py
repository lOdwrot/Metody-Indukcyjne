import voting
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

def normal(_df, _parts, _k, _distanceMetric, votingMethod):

    mData = _df

    featureVals = [x for x in mData if x != 'Class']
    # if _stratified == False:
    #     kf = KFold(n_splits=_parts, shuffle=_shuffle)
    # else:
    kf = StratifiedKFold(n_splits=_parts)

    allTargets = []
    allPreds = []
    # if _stratified == False:
    #     for train_index, test_index in kf.split(mData):
    #         tarinSet = mData.iloc[train_index]

    #         examSet = mData.iloc[test_index]
    #         examTargets = mData.iloc[test_index]['Class']

    #         if _classiFierType == 'gausian':
    #             mClassifier = naiveBayes.getClassifireGausian(tarinSet)
    #         else:
    #             mClassifier = naiveBayes.getClassifireMultinomial(tarinSet)

    #         predTargets = naiveBayes.predict(mClassifier, examSet)

    #         allTargets.extend(examTargets)
    #         allPreds.extend(predTargets)
    # else:
    for train_index, test_index in kf.split(mData, mData['Class']):
        tarinSet = mData.iloc[train_index]

        examSet = mData.iloc[test_index]
        examTargets = mData.iloc[test_index]['Class']

        if votingMethod == 'squaredDistances':
            mClassifier = KNeighborsClassifier(n_neighbors=_k, weights=voting.squaredDistances, metric=_distanceMetric)
        else:
            mClassifier = KNeighborsClassifier(n_neighbors=_k, weights=votingMethod, metric=_distanceMetric)

        mClassifier.fit(tarinSet[featureVals], tarinSet['Class'])
        predTargets = mClassifier.predict(examSet[featureVals])

        allTargets.extend(examTargets)
        allPreds.extend(predTargets)

    return allTargets, allPreds
        
        
