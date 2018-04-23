import io
import requests
import pandas as pd

diabetesDataPath = './dataSets/diabetes/data.txt'   

def getDataSetWine():
    mUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    s=requests.get(mUrl).content
    realNames=[
            'Class',
            'Malic acid',
            'Ash',
            'Alcalinity of ash  ',
            'Magnesium',
            'Total phenols',
            'Flavanoids',
            'Nonflavanoid phenols',
            'Proanthocyanins',
            'olor intensity',
            'ue',
            'D280/OD315 of diluted wines',
            'roline'
            ]
    useNamse = [
                'Class',
                'f1',
                'f2',
                'f3',
                'f4',
                'f5',
                'f6',
                'f7',
                'f8',
                'f9',
                'f10',
                'f11',
                'f12',
                'f13'
            ]


    c=pd.read_csv(io.StringIO(s.decode('utf-8')),
                    sep=",", 
                    names=useNamse)

    return c

def getDataSetGlass():
    mUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    s=requests.get(mUrl).content
    useNamse = [
                'index',
                'f1',
                'f2',
                'f3',
                'f4',
                'f5',
                'f6',
                'f7',
                'f8',
                'f9',
                'Class'
            ]


    c=pd.read_csv(io.StringIO(s.decode('utf-8')),
                    sep=",", 
                    names=useNamse)
    del c['index']
    return c

def getDataSetDiabetes():
    mUrl = "http://web.archive.org/web/20161120103553/http://archive.ics.uci.edu:80/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    s=requests.get(mUrl).content
    useNamse = [
                'f1',
                'f2',
                'f3',
                'f4',
                'f5',
                'f6',
                'f7',
                'f8',
                'Class'
            ]


    c=pd.read_csv(io.StringIO(s.decode('utf-8')),
                    sep=",", 
                    names=useNamse)
    return c



def getDataSetIris():
    mUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    s=requests.get(mUrl).content
    realNames=[
                'sepal length in cm',
                'sepal width in cm',
                'petal length in cm',
                'petal width in cm',
                'Class'
            ]

    useNamse = [
                'f1',
                'f2',
                'f3',
                'f4',
                'Class'
            ]


    c=pd.read_csv(io.StringIO(s.decode('utf-8')),
                    sep=",", 
                    names=useNamse)

def getDataSetKnowlegde():
    mUrl = "knowledge.txt"
    with open(mUrl, 'r') as myfile:
        s=myfile.read()
   
    realNames=[
                'sepal length in cm',
                'sepal width in cm',
                'petal length in cm',
                'petal width in cm',
                'Class'
            ]

    useNamse = [
                'STG',
                'SCG',
                'STR',
                'LPR',
                'PEG',
                'Class'
            ]


    c=pd.read_csv(io.StringIO(s),
                    sep=",", 
                    names=useNamse)

    return c
