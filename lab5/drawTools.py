import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools


def drawHistDiagonal(_df):
  uniqClasses = set(_df['Class'])
  drawVals = [x for x in _df if x != 'Class']
  print(drawVals)
  sns.pairplot(_df, hue="Class", vars=drawVals, markers='+')


def drawConfusionMatrix(y_True, y_Pred, classes=None, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, pltNumb=None):
    if pltNumb is None:
        fig = plt.figure()
    else:
        plt.subplot(2,3,pltNumb)

    cm = confusion_matrix(y_True, y_Pred)
    if classes is None:
        classes = set(y_True)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')


def drawBars(_df):
    print(_df)
    fig, ax = plt.subplots()
    featureVals = [x for x in _df if x != 'class']
    colors = ['r', 'g', 'b', 'c', 'm', 'r']
    baseWidth = 0.15

    ax.set_ylim(
        min([min(_df[x]) for x in featureVals]) - 0.01, 
        max([max(_df[x]) for x in featureVals]) + 0.01,
        )

    mClasses = _df['class']
    
    for xIndex, xVal in enumerate(featureVals):
        for yIndex, yVal in enumerate(_df[xVal]):
            ax.bar(yIndex * baseWidth + xIndex, yVal, color=colors[yIndex], width=baseWidth, label=mClasses[yIndex] if xIndex == 0 else "")

    plt.xticks([x + 2.5 * baseWidth for x in range(0, len(featureVals))], featureVals)
    plt.legend()

    # indexes = list(range(0, len(featureVals)))
    # for index, val in enumerate(_barVals):
    #     nIndexes = [x + index*baseWidth for x in indexes]
    #     if _legendLabels is None:
    #         ax.bar(nIndexes, val, color=colors[index%(len(colors)-1)], width=baseWidth)
    #     else:
    #         ax.bar(nIndexes, val, color=colors[index%(len(colors)-1)], width=baseWidth, label=_legendLabels[index])
    #         plt.legend()

    # sns.barplot(data=_df, hue="class", y=["fscore", "accuracy"], x='class')
  

