library(data.table)

getDataSetIris <- function() {
  mUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
  realNames=c(
    'sepal length in cm',
    'sepal width in cm',
    'petal length in cm',
    'petal width in cm',
    'Class'
  )
  useNames <- c(
    'f1',
    'f2',
    'f3',
    'f4',
    'Class'
  )
  data <- fread(mUrl)
  data[data=="Iris-setosa"] <- 1
  data[data=="Iris-versicolor"] <- 2
  data[data=="Iris-virginica"] <- 3
  
  names(data) <- useNames
  
  resultDF = as.data.frame(data)
  resultDF['Class'] <- sapply(resultDF['Class'], as.numeric)
  
  return(range01(resultDF))
}

getDataSetWine <- function() {
  mUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
  realNames=c(
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
  )
  useNames <- c(
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
  )
  
  data <- fread(mUrl)
  names(data) <- useNames
  
  resultDF = as.data.frame(data)
  return(range01(resultDF))
}

getDataSetDiabetes <- function() {
  mUrl = "./dataSets/diabetes.txt"
  useNames <- c(
    'f1',
    'f2',
    'f3',
    'f4',
    'f5',
    'f6',
    'f7',
    'f8',
    'Class'
  )
  
  data <- fread(mUrl)
  names(data) <- useNames
  
  resultDF = as.data.frame(data)
  return(range01(resultDF))
}

getDataSetKnowledge <- function() {
  mUrl = "./dataSets/knowledge.txt"
  useNames <- c(
    'STG',
    'SCG',
    'STR',
    'LPR',
    'PEG',
    'Class'
  )
  
  # 1 very low
  # 2 low
  # 3 middle
  # 4 high
  
  data <- fread(mUrl)
  names(data) <- useNames
  
  resultDF = as.data.frame(data)
  return(range01(resultDF))
}

getDataSetGlass <- function() {
  mUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
  useNames <- c(
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
  )
  
  data <- fread(mUrl)
  names(data) <- useNames
  
  data <- subset(data, select = -c(1))
  
  resultDF = as.data.frame(data)
  return(range01(resultDF))
}


range01 <- function(mDF) {
  mFatuerNames <- colnames(mDF)[colnames(mDF) != 'Class']
  result = data.frame()
  for(n in mFatuerNames) {
    x = mDF[n]
    mDF[n] = (x-min(x))/(max(x)-min(x))
  }
  return(mDF)
}