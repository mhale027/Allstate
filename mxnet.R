library(data.table)
library(xgboost)
library(Matrix)
library(MASS)
library(mxnet)

setwd("~/projects/kaggle/allstate")
source("features.R")
load("alldata.nzv.rip.RData")

features <- 2:101
response <- 1

train.x = data.matrix(alldata[f1, -1])
train.y = alldata[f1, 1]
valid.x <- data.matrix(alldata[fold1, -1])
valid.y <- alldata[fold1, 1]
test.x = data.matrix(alldata[te, -1])
test.y = alldata[te, 1]


data <- mx.symbol.Variable("data")
l1 <- mx.symbol.FullyConnected(data, name = "l1", num.hidden = 400)
a1 <- mx.symbol.Activation(l1, name = "a1", act_type = 'relu')
d1 <- mx.symbol.Dropout(a1, name = 'd1', p = 0.4)
l2 <- mx.symbol.FullyConnected(d1, name = "l2", num.hidden = 200)
a2 <- mx.symbol.Activation(l2, name = "a2", act_type = 'relu')
d2 <- mx.symbol.Dropout(a2, name = 'd2', p = 0.2)
l3 <- mx.symbol.FullyConnected(d2, name = "l3", num.hidden = 1)
lro <- mx.symbol.MAERegressionOutput(l3, name = "lro")

mx.set.seed(0)
model <- mx.model.FeedForward.create(
      lro, X=t(train.x), y=train.y,
      eval.data=list(data=t(valid.x), label=valid.y),
      ctx=mx.cpu(), num.round=10, array.batch.size=100,
      learning.rate=.001, momentum=0.9, array.layout = 'colmajor', eval.metric=mx.metric.mae)

preds <- as.numeric(predict(model, test.x))


# train mae: .41655  100 vars shift: 200;   LB: 1237


