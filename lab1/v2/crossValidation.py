import naiveBayes
import discretization
from sklearn.model_selection import KFold, StratifiedKFold

def normal(_df, _parts, _classiFierType, _discretization=None, _shuffle=False, _stratified=False):

    if _discretization is None:
        mData = _df
    elif _discretization == 'width':
        mData = discretization.equalWidth(_df)
    elif _discretization == 'freq':
        mData = discretization.equalFreq(_df)
    elif _discretization == 'mdpl':
        mData = discretization.discMdlp(_df)

    featureVals = [x for x in mData if x != 'Class']
    if _stratified == False:
        kf = KFold(n_splits=_parts, shuffle=_shuffle)
    else:
        kf = StratifiedKFold(n_splits=_parts)

    allTargets = []
    allPreds = []
    if _stratified == False:
        for train_index, test_index in kf.split(mData):
            tarinSet = mData.iloc[train_index]

            examSet = mData.iloc[test_index]
            examTargets = mData.iloc[test_index]['Class']

            if _classiFierType == 'gausian':
                mClassifier = naiveBayes.getClassifireGausian(tarinSet)
            else:
                mClassifier = naiveBayes.getClassifireMultinomial(tarinSet)

            predTargets = naiveBayes.predict(mClassifier, examSet)

            allTargets.extend(examTargets)
            allPreds.extend(predTargets)
    else:
        for train_index, test_index in kf.split(mData, mData['Class']):
            tarinSet = mData.iloc[train_index]

            examSet = mData.iloc[test_index]
            examTargets = mData.iloc[test_index]['Class']

            if _classiFierType == 'gausian':
                mClassifier = naiveBayes.getClassifireGausian(tarinSet)
            else:
                mClassifier = naiveBayes.getClassifireMultinomial(tarinSet)

            predTargets = naiveBayes.predict(mClassifier, examSet)

            allTargets.extend(examTargets)
            allPreds.extend(predTargets)

    return allTargets, allPreds
        
        
