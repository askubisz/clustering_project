library(pdfCluster)

set.seed(0)

# Load datasets and labels

# Iris
data_iris=iris[-5] # Remove the column with the label
labels_iris=match(iris$Species, unique(iris$Species))
data_iris=as.data.frame(scale(data_iris,center=TRUE, scale=TRUE))

# Ecoli
data_ecoli=read.table("C:/Users/askub/OneDrive/Pulpit/Uni/Datasci/400dissertation/data/clustbench/uci/ecoli.data.gz",
                      header = FALSE,
                      sep = "",
                      dec = ".")
labels_ecoli=read.table("C:/Users/askub/OneDrive/Pulpit/Uni/Datasci/400dissertation/data/clustbench/uci/ecoli.labels0.gz",
                        header = FALSE,
                        sep = "",
                        dec = ".")
labels_ecoli=labels_ecoli$V1
data_ecoli=as.data.frame(scale(data_ecoli,center=TRUE, scale=TRUE))

# Glass
data_glass=read.table("C:/Users/askub/OneDrive/Pulpit/Uni/Datasci/400dissertation/data/clustbench/uci/glass.data.gz",
                      header = FALSE,
                      sep = "",
                      dec = ".")
labels_glass=read.table("C:/Users/askub/OneDrive/Pulpit/Uni/Datasci/400dissertation/data/clustbench/uci/glass.labels0.gz",
                        header = FALSE,
                        sep = "",
                        dec = ".")
labels_glass=labels_glass$V1
data_glass=as.data.frame(scale(data_glass,center=TRUE, scale=TRUE))

# Wine
data_wine=read.table("C:/Users/askub/OneDrive/Pulpit/Uni/Datasci/400dissertation/data/clustbench/uci/wine.data.gz",
                      header = FALSE,
                      sep = "",
                      dec = ".")
labels_wine=read.table("C:/Users/askub/OneDrive/Pulpit/Uni/Datasci/400dissertation/data/clustbench/uci/wine.labels0.gz",
                        header = FALSE,
                        sep = "",
                        dec = ".")
labels_wine=labels_wine$V1
data_wine=as.data.frame(scale(data_wine,center=TRUE, scale=TRUE))

# Yeast
data_yeast=read.table("C:/Users/askub/OneDrive/Pulpit/Uni/Datasci/400dissertation/data/clustbench/uci/yeast.data.gz",
                     header = FALSE,
                     sep = "",
                     dec = ".")
labels_yeast=read.table("C:/Users/askub/OneDrive/Pulpit/Uni/Datasci/400dissertation/data/clustbench/uci/yeast.labels0.gz",
                       header = FALSE,
                       sep = "",
                       dec = ".")
labels_yeast=labels_yeast$V1
data_yeast=as.data.frame(scale(data_yeast,center=TRUE, scale=TRUE))

# Fit models and print AR 

iris_lloyd=kmeans(x=data_iris, centers=3, nstart=100, iter.max=100, algorithm="Lloyd")
iris_macqueen=kmeans(x=data_iris, centers=3, nstart=100, iter.max=100, algorithm="MacQueen")
iris_hartigan=kmeans(x=data_iris, centers=3, nstart=100, iter.max=100, algorithm="Hartigan-Wong")
print(c(adj.rand.index(iris_lloyd$cluster, labels_iris), adj.rand.index(iris_macqueen$cluster, labels_iris), adj.rand.index(iris_hartigan$cluster, labels_iris)))


ecoli_lloyd=kmeans(x=data_ecoli, centers=8, nstart=100, iter.max=100, algorithm="Lloyd")
ecoli_macqueen=kmeans(x=data_ecoli, centers=8, nstart=100, iter.max=100, algorithm="MacQueen")
ecoli_hartigan=kmeans(x=data_ecoli, centers=8, nstart=100, iter.max=100, algorithm="Hartigan-Wong")
print(c(adj.rand.index(ecoli_lloyd$cluster, labels_ecoli), adj.rand.index(ecoli_macqueen$cluster, labels_ecoli), adj.rand.index(ecoli_hartigan$cluster, labels_ecoli)))


glass_lloyd=kmeans(x=data_glass, centers=6, nstart=100, iter.max=100, algorithm="Lloyd")
glass_macqueen=kmeans(x=data_glass, centers=6, nstart=100, iter.max=100, algorithm="MacQueen")
glass_hartigan=kmeans(x=data_glass, centers=6, nstart=100, iter.max=100, algorithm="Hartigan-Wong")
print(c(adj.rand.index(glass_lloyd$cluster, labels_glass), adj.rand.index(glass_macqueen$cluster, labels_glass), adj.rand.index(glass_hartigan$cluster, labels_glass)))


wine_lloyd=kmeans(x=data_wine, centers=3, nstart=100, iter.max=100, algorithm="Lloyd")
wine_macqueen=kmeans(x=data_wine, centers=3, nstart=100, iter.max=100, algorithm="MacQueen")
wine_hartigan=kmeans(x=data_wine, centers=3, nstart=100, iter.max=100, algorithm="Hartigan-Wong")
print(c(adj.rand.index(wine_lloyd$cluster, labels_wine), adj.rand.index(wine_macqueen$cluster, labels_wine), adj.rand.index(wine_hartigan$cluster, labels_wine)))


yeast_lloyd=kmeans(x=data_yeast, centers=10, nstart=100, iter.max=100, algorithm="Lloyd")
yeast_macqueen=kmeans(x=data_yeast, centers=10, nstart=100, iter.max=100, algorithm="MacQueen")
yeast_hartigan=kmeans(x=data_yeast, centers=10, nstart=100, iter.max=100, algorithm="Hartigan-Wong")
print(c(adj.rand.index(yeast_lloyd$cluster, labels_yeast), adj.rand.index(yeast_macqueen$cluster, labels_yeast), adj.rand.index(yeast_hartigan$cluster, labels_yeast)))
