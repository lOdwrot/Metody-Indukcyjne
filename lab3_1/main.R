source("./dataLoader.R")
source("./statsCounter.R")
# library(ggplot2)
library(ggplot2)
library(GGally)
library(datasets)
library(NMF)
library(clusteval)
library (vegan)


# Test first data set

clustersQuantity = c(2, 3, 6, 10, 15, 20, 30, 50)


mDataSet <- getDataSetIris()
mDataSetName <- 'Iris'
mFeatureColumns = colnames(mDataSet)[colnames(mDataSet) != 'Class']
set.seed(20)

print(mDataSetName)
print('Kmean ')
for(cq in clustersQuantity) {
  kmeanCluster <- kmeans(mDataSet[mFeatureColumns], cq, nstart = 20, iter.max = 20)
  kmeanStats <- countCluesteringStats(mDataSet, kmeanCluster$cluster)
  
  sprintf('cq# %d ## %.3f ## %.3f ## %.3f ## %.3f end#', cq, kmeanStats$purity, kmeanStats$rand, kmeanStats$dunn, kmeanStats$dbi)
}

print('Pam Stats ')
for(cq in clustersQuantity) {
  pamClaster <- pam(mDataSet[mFeatureColumns], 3)
  pamStats <- countCluesteringStats(mDataSet, pamClaster$cluster)
  
  sprintf('cq# %d ## %.3f ## %.3f ## %.3f ## %.3f end#', cq, pamClaster$purity, pamClaster$rand, pamClaster$dunn, pamClaster$dbi)
}



# print(irisCluster)
# 
# # 
# 
# print(stats)
# countCluesteringStats(mDataSet, pamClaster$cluster, TRUE)



# a <- cluster_similarity(irisCluster$cluster, pamClaster$clustering,
#                    similarity = "rand",
#                    method = "independence")


# irisCluster$cluster <- as.factor(irisCluster$cluster)
# ggplot(mData, aes(f3, f4, color = Class))  + geom_point

# ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point()
# ggplot(mDataSet, aes(f1,f2, color = Class)) + geom_point()



# Ekstra rzeczy

# ggpairs(mDataSet, aes(color = Class))
