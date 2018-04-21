# Stratyfikowalna cross-walidacja
require(caret)
require(dplyr)
source("./dataLoader.R")
source("./statsCounter.R")
library(C50)


stratifiedCrossC50 <- function(pData, pTreeParams, foldsQuantity = 10) {
  mFeatureColumns = colnames(pData)[colnames(pData) != 'Class']
  
  print('mFeatureColumns')
  print(mFeatureColumns)
  
  # Tworzysz k zbioróW na podstawie klas, które są w wektorze iris$Spiecies
  folds <- createFolds(pData$Class, k = foldsQuantity, list = TRUE, returnTrain = FALSE)
  
  vecPredicted <- vector(length = length(folds), mode = 'list')
  vecTrue <- vector(length = length(folds), mode = 'list')
  
  # Każdy podzbiór fold w liście folds zawiera indeksy do tabeli testowej
  index = 1
  for(fold in folds) {
    testData <- pData[fold, ] # wszystkie wiersze testowe
    trainData <- pData[-fold, ] # wszystkie wiersze poza testowymi - czyli treningowe
    
    cTree <- C5.0(x = trainData[, mFeatureColumns], y = as.factor(trainData$Class), control=pTreeParams)
    
    predictedClasses <- predict( cTree, testData[, mFeatureColumns], type="class")
    
    
    vecPredicted[[index]] <- predictedClasses
    vecTrue[[index]] <- testData[, 'Class']
    
    index = index + 1
  }
  
  vecPredicted <- unlist(vecPredicted)
  vecTrue <- unlist(vecTrue)
  
  predictionDf = data.frame(vecPredicted=vecPredicted, vecTrue=vecTrue)
  return (countStats(predictionDf))
}


