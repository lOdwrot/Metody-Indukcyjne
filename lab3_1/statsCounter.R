library(NMF)
require(caret)
library(clValid)
library(fossil)
library(pdfCluster)
library (vegan)
library(clusterSim)

countCluesteringStats <- function(mDataSet, clusteredData) {
  mFatuerNames <- colnames(mDataSet)[colnames(mDataSet) != 'Class']
  # print('Purity')
  mPurity <- purity(as.factor(clusteredData), mDataSet$Class)
  # print(mPurity)
  # 
  # print('Rand')
  mRand <- rand.index(clusteredData, mDataSet$Class)
  # print(mRand)

  # print('Dunn index')
  mDunn <- dunn(cluster=clusteredData, method="euclidean", Data=mDataSet[mFatuerNames])
  # print(mDunn)
  
  # print('DBI')
  mDbi <- index.DB(mDataSet[mFatuerNames], clusteredData)$DB
  # print(mDbi)
  
  return (list(
    purity=mPurity,
    rand=mRand,
    dunn=mDunn,
    dbi=mDbi
  ))

}

countStats <- function(mDf) {
  print('Counting stats')
  cm <- confusionMatrix(mDf[,'vecPredicted'] , mDf[,'vecTrue'], mode="prec_recall")
  
  
  if(!is.null(nrow(cm$byClass))) {
    f1Vcet <- cm$byClass[,"F1"]
    recallVect <- cm$byClass[,"Recall"]
    precisionVect <- cm$byClass[,"Precision"]
  } else {
    f1Vcet <- cm$byClass["F1"]
    recallVect <- cm$byClass["Recall"]
    precisionVect <- cm$byClass["Precision"]
  }
  
  
  f1Vcet[is.nan(f1Vcet)] = 0
  f1Vcet[is.na(f1Vcet)] = 0
  
  recallVect[is.nan(recallVect)] = 0
  recallVect[is.na(recallVect)] = 0
  
  precisionVect[is.nan(precisionVect)] = 0
  precisionVect[is.na(precisionVect)] = 0
  
  # count means
  if(!is.null(nrow(cm$byClass))) {
    classesNames <- unique(mDf[,'vecTrue'])
    for(x in seq(length(classesNames))) {
      occurances <- sum(mDf[,'vecTrue'] == classesNames[x])
      f1Vcet[[x]] = f1Vcet[[x]] * occurances * length(classesNames) / length(mDf[,'vecTrue'])
      recallVect[[x]] = f1Vcet[[x]] * occurances * length(classesNames) / length(mDf[,'vecTrue'])
      precisionVect[[x]] = f1Vcet[[x]] * occurances * length(classesNames) / length(mDf[,'vecTrue'])
    }
  }
  
  
  return(list(
    F1=mean(f1Vcet),
    Recall=mean(recallVect),
    Precision=mean(precisionVect),
    Accuracy=cm$overall["Accuracy"],
    cmTable=cm$table,
    fullCmByClass=cm$byClass
  ))
}