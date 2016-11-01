library(data.table)
library(randomForest)
library(ParallelForest)
library(ranger)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(Metrics)
library(nnet)
library(h2o)


load("data.RData")
load("features.RData")

sample <- data.frame(fread("sample_submission.csv", header = TRUE))

lengths <- features[[2]]
training.length <- lengths[1]
test.length <- lengths[2]

tr <- 1:training.length
te <- (1+training.length):(training.length + test.length)

ids <- features[[3]]
training.id <- ids[tr]
test.id <- ids[te]

losses <- features[[4]]
training.loss <- losses[tr]


x.vars <- names(alldata)[-1]
response <- names(alldata)[1]


train <- alldata[tr,]


h2o.init(nthread = -1)

train <- as.h2o(train)
test <- as.h2o(alldata[te,x.vars])

dnn.1 <- h2o.deeplearning(x.vars, response, train,
                          hidden = c(50, 10),
                          seed = 7890,
                          epochs = 10,
                          nfold = 3
                          )

pred.dnn.1 <- as.numeric(unlist(as.data.frame(predict(dnn.1, train))))
pred.test.dnn.1 <- as.numeric(unlist(as.data.frame(predict(dnn.1, test))))



train <- cbind(alldata[tr,], pred.dnn.1)
test <- cbind(alldata[tr,] pred.test.dnn.1])

x.vars <- names(train)[-1]

train <- as.h2o(train)
test <- as.h2o(test)

gc()

dnn.2 <- h2o.deeplearning(x.vars, response, train,
                          hidden = c(150, 30),
                          seed = 7890,
                          epochs = 10,
                          nfold = 4
)

pred.dnn.2 <- as.numeric(unlist(as.data.frame(predict(dnn.2, train))))
pred.test.dnn.2 <- as.numeric(unlist(as.data.frame(predict(dnn.2, test))))






train <- cbind(alldata[tr,], pred.dnn.1, pred.dnn.2)
test <- cbind(alldata[tr,] pred.test.dnn.1, pred.test.dnn.2)

x.vars <- names(train)[-1]

train <- as.h2o(train)
test <- as.h2o(test)

gc()

dnn.3 <- h2o.deeplearning(x.vars, response, train,
                          hidden = c(200, 100),
                          seed = 7890,
                          epochs = 10,
                          nfold = 5
)

pred.dnn.3 <- as.numeric(unlist(as.data.frame(predict(dnn.3, train))))
pred.test.dnn.3 <- as.numeric(unlist(as.data.frame(predict(dnn.3, test))))






h2o.shutdown()


pred <- (pred.test.dnn.1 + pred.test.dnn.2 + pred.test.dnn.3)/3
sample$loss <- pred





write.csv(sample,'submission.10.31.1.csv',row.names = FALSE)






