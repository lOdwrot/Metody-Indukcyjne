from sklearn.naive_bayes import GaussianNB, MultinomialNB

def getClassifireGausian(_df):
    featureVals = [x for x in _df if x != 'Class']
    gnb = GaussianNB()
    gnb.fit(_df[featureVals], _df['Class'])
    return gnb

def getClassifireMultinomial(_df):
    featureVals = [x for x in _df if x != 'Class']
    gnb = MultinomialNB(alpha=1.0)
    gnb.fit(_df[featureVals], _df['Class'])
    return gnb


def predict(_classifier, _df):
    featureVals = [x for x in _df if x != 'Class']
    y_pred = _classifier.predict(_df[featureVals])
    return y_pred