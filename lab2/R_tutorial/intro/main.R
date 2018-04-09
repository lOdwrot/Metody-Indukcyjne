source("./dataLoader.R")
source("./crossvalidationStratified.R")
source("./crossValidationNormal.R")
library(C50)


mTreeParams <- C5.0Control(CF=0.25, noGlobalPruning=TRUE)


# Test first data set
mDataSet <- getDataSetWine()
mFeatureColumns = colnames(mDataSet)[colnames(mDataSet) != 'Class']



# creating and drawing tree
cTree <- C5.0(x = mDataSet[, mFeatureColumns], y = as.factor(mDataSet$Class), control=mTreeParams)
print(cTree)
# plot(cTree)



# test min cases 

vecMinCases = c(2)
vecMinCasesFscore <- seq(length(vecMinCases))
  for(x in seq(length(vecMinCases))){
    mTreeParams <- C5.0Control(minCases = vecMinCases[x])
    mResult <- stratifiedCrossC50(mDataSet, mTreeParams, 9)
    vecMinCasesFscore[x] <- mResult["F1"] 
  }

print(vecMinCasesFscore)
vecMinCasesFscore <- unlist(vecMinCasesFscore)
barplot(vecMinCasesFscore, names.arg=vecMinCases, ylim = c(0,1))


# 
# mResult <- stratifiedCrossC50(mDataSet, mTreeParams)
# print(mResult)
# test crossvalidation

testedFoldSizes <- c(2,3,4,5,6,7,8,9,10)
mData <- getDataSetDiabetes()

vecStratified <- vector(length = length(testedFoldSizes), mode = 'list')
vecNormal <- vector(length = length(testedFoldSizes), mode = 'list')
vecNormalRandomize <- vector(length = length(testedFoldSizes), mode = 'list')


for(x in seq(length(testedFoldSizes))) {
  vecStratified[x] <- stratifiedCrossC50(mData, C5.0Control(), testedFoldSizes[x])
  vecNormal[x] <- normalCrossC50(mData, C5.0Control(), testedFoldSizes[x], FALSE)
  vecNormalRandomize[x] <- normalCrossC50(mData, C5.0Control(), testedFoldSizes[x], TRUE)
}

dt <- data.table(testedFoldSizes, vecStratified, vecNormal, vecNormalRandomize)
names(dt) <- c("Folds", "Stratified crossvalidation", "Normall crossvalidation", "Normall randomize crossvalidation")
print(dt)

