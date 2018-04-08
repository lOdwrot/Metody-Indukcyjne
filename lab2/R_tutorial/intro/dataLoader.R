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
  names(data) <- useNames
  return(as.data.frame(data))
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
  
  return(as.data.frame(data))
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
  
  return(as.data.frame(data))
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
  
  return(as.data.frame(data))
}