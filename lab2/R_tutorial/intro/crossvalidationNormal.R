# Zwykła cross validacja
normalCrossC50 <- function(pData, pTreeParams, foldsQuantity = 10, randomize = TRUE) {
  mFeatureColumns = colnames(pData)[colnames(pData) != 'Class']
  
  print('mFeatureColumns')
  print(mFeatureColumns)
  
  # wektor indeksów
  x <- seq_len(nrow(pData))
  if (randomize == TRUE) {
    x <- sample(x, length(x), replace = FALSE) # shuffle
  }
  
  
  
  # Podział na części
  folds <- cut(x, breaks = foldsQuantity, labels = FALSE)
  # Podział na foldy
  folds <- lapply(seq_len(foldsQuantity), function(i) which(folds == i))
  
  vecPredicted <- vector(length = length(folds), mode = 'list')
  vecTrue <- vector(length = length(folds), mode = 'list')
  
  
  
  index = 1
  for(fold in folds) {
    testData <- pData[fold, ] # wszystkie wiersze testowe
    trainData <- pData[-fold, ] # wszystkie wiersze treningowe

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