source("./dataLoader.R")
source("./crossvalidationStratified.R")
source("./crossValidationNormal.R")
# library(ggplot2)
library(ggplot2)
library(datasets)
library(NMF)
library(clusteval)

# Test first data set
mDataSet <- getDataSetIris()
mFeatureColumns = colnames(mDataSet)[colnames(mDataSet) != 'Class']




# ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point()
# ggplot(mDataSet, aes(f1,f2, color = Class)) + geom_point()

set.seed(20)
irisCluster <- kmeans(mDataSet[mFeatureColumns], 3, nstart = 20)
pamClaster <- pam(mDataSet[mFeatureColumns], 10)
print(irisCluster)
# 
countCluesteringStats(mDataSet, irisCluster$cluster)

# ggpairs(mDataSet)


# a <- cluster_similarity(irisCluster$cluster, pamClaster$clustering,
#                    similarity = "rand",
#                    method = "independence")


# irisCluster$cluster <- as.factor(irisCluster$cluster)
# ggplot(mData, aes(f3, f4, color = Class))  + geom_point()