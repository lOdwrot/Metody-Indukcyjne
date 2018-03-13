import numpy as np
from sklearn.preprocessing import quantile_transform
import Orange

def discretizationEqualWidth(_data, _parts):
    print('Equal width')
    minMaxVals = np.array([])
    transposedData = np.array(_data[:]).transpose()
    resultData = []

    for featureVector in transposedData:
        resultData.append(
            np.digitize(
                featureVector,
                np.linspace(min(featureVector), max(featureVector), _parts)
            )
        )

    resultData = np.array(resultData).transpose()
    return resultData
    
    
def discretizationQuantileTransform(_data, _quantiles=100):
    print('Quantile')
    nData = list(_data)
    return quantile_transform(nData, n_quantiles=_quantiles, random_state=0) 

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def discretizationEqualFreq(_data):
    print('Equal freq')
    groups = 15
    orangeTable = Orange.data.Table(_data)
    disc = Orange.preprocess.Discretize()
    disc.method = Orange.preprocess.discretize.EqualFreq(n=groups)
    orangeDiscrete = disc(orangeTable)
    mSets = [set() for i in range(0, len(_data[0]))]
    for sample in orangeDiscrete:
        for index in range(0, len(sample)):
            numbers = [float(s) for s in str(sample[index]).split() if is_number(s)]
            for numb in numbers:
                mSets[index].add(numb)

    # create bins
    mBins = []
    for mSet in mSets:
        mBins.append(sorted(list(mSet)))


    transposedData = np.array(_data[:]).transpose()
    resultData = []
    for index, featureVector in enumerate(transposedData):
        resultData.append(
            np.digitize(
                featureVector,
                mBins[index]
            )
        )

    resultData = np.array(resultData).transpose()
    return resultData