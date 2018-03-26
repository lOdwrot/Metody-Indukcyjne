import numpy as np
from sklearn.preprocessing import quantile_transform
import pandas as pd
from mdlp.discretization import MDLP

def equalWidth(_df):
    featureVals = [x for x in _df if x != 'Class']
    discretizedMap = {'Class': _df['Class']}

    for x in featureVals:
        discretizedMap[x] = pd.cut(_df[x], 5, labels=False)

    nFrame = pd.DataFrame.from_dict(discretizedMap)
    return nFrame
    
    
def discMdlp(_df):
    featureVals = [x for x in _df if x != 'Class']
    transformer = MDLP()
    discretizedMap = {'Class': _df['Class']}

    discret = transformer.fit_transform(_df[featureVals],_df['Class'])
    nFrame = pd.DataFrame(data=discret, columns=featureVals)
    nFrame.loc[:,'Class'] = pd.Series(_df['Class'])
    
    return nFrame

def equalFreq(_df):
    featureVals = [x for x in _df if x != 'Class']
    discretizedMap = {'Class': _df['Class']}

    for x in featureVals:
        discretizedMap[x] = pd.qcut(_df[x], 5, labels=False, duplicates='drop')

    nFrame = pd.DataFrame.from_dict(discretizedMap)
    return nFrame