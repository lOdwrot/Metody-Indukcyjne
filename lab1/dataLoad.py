diabetesDataPath = './dataSets/diabetes/data.txt'

def testFun():
    print('Hello')
# from file


def getDiabetesDataSet():
    features = []
    targets = []
    featureDesc = [
        'Number of times pregnant',
        'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
        'Diastolic blood pressure (mm Hg)',
        'Triceps skin fold thickness (mm)',
        '2-Hour serum insulin (mu U/ml)',
        'Body mass index (weight in kg/(height in m)^2)',
        'Diabetes pedigree function',
        'Age (years)'
    ]
    targetNames = [
        'positive',
        'negative'
    ]

    with open(diabetesDataPath, "r") as ins:
        for line in ins:
            lineData = [float(x) for x in line.replace('\n', '').split(',')]
            targets.append(int(lineData.pop(len(lineData) - 1)))
            features.append(lineData)

    return {
        "data": features,
        "target": targets,
        "target_names": targetNames,
        "featureDesc": featureDesc
    }
