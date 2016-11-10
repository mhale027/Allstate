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

setwd("~/Projects/kaggle/Allstate")

set.seed(7890)

load("df.RData")
load("tr.RData")
load("te.RData")
load("training.loss.RData")

sample <- data.frame(fread("sample_submission.csv", header = TRUE))

mean.loss <- mean(training.loss)




ytrain <- training.loss
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = data.matrix(ytrain))
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))


xgb.grid <- expand.grid(
      nrounds = 10, 
      eta = 0.2, 
      max_depth = c(6, 8, 10, 12, 14, 16), 
      gamma = c(0, 3, 5), 
      colsample_bytree = c(0.6, 0.7, 0.85, 1.0), 
      min_child_weight = c(0, 1, 2, 3)
)

xgb.train.control <- trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      allowParallel = TRUE
)


xgb.prep <- train(x=xtrain[1:1000,],
                  y=ytrain[1:1000],
                  trControl = xgb.train.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree"
)



xgb.1.max_depth <- xgb.prep$bestTune$max_depth
xgb.1.gamma <- xgb.prep$bestTune$gamma
xgb.1.colsample_bytree <- xgb.prep$bestTune$colsample_bytree
xgb.1.min_child_weight <- xgb.prep$bestTune$min_child_weight

xgb.1.params <- list(
      objective = "reg:linear",
      eta = 0.2,              
      max_depth = 12, 
      gamma = 5,
      colsample_bytree = .8,
      min_child_weight = 1,
      eval_metric = "rmse"
)


xgb.1 <- xgboost(xtrain, params = xgb.1.params, nfold = 4, nrounds = 300, early_stopping_rounds = 5)

pred.test.xgb.1 <- predict(xgb.1, xtest)

head(pred.test.xgb.1)
#1871.790 2246.508 5439.477 5300.870 1137.884 2224.441
#1635.1407 1124.9293 6663.1151 6018.9569  380.7898 3261.7621           LB: 1239.73412,



xgb.grid.2 <- expand.grid(
      nrounds = 10, 
      eta = 0.01, 
      max_depth = c(11, 12, 13), 
      gamma = c(4, 5, 7), 
      colsample_bytree = c(0.8, 0.85, .9), 
      min_child_weight = c(2.5, 3, 3.5)
)

xgb.train.control.2 <- trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      allowParallel = TRUE
)


xgb.prep.2 <- train(x=xtrain[1:1000,],
                  y=ytrain[1:1000],
                  trControl = xgb.train.control.2,
                  tuneGrid = xgb.grid.2,
                  method = "xgbTree"
)



xgb.2.max_depth <- xgb.prep.2$bestTune$max_depth
xgb.2.gamma <- xgb.prep.2$bestTune$gamma
xgb.2.colsample_bytree <- xgb.prep.2$bestTune$colsample_bytree
xgb.2.min_child_weight <- xgb.prep.2$bestTune$min_child_weight


xgb.2.params <- list(
      objective = "reg:linear",
      eta = 0.01,              
      max_depth = xgb.1.max_depth, 
      gamma = xgb.1.gamma,
      colsample_bytree = xgb.1.colsample_bytree,
      min_child_weight = xgb.1.min_child_weight,
      eval_metric = "rmse"      
)
      
xgb.2 <- xgboost(data = train, early_stopping_rounds = 5, nrounds = 500, params = xgb.2.params)

pred.test.xgb.2 <- predict(xgb.2, test)
# 1826.741 2318.976 6303.412 6218.342 1014.514 2447.794
# 1774.2902 2144.4787 5976.2269 6660.6348  952.4483 2407.0292          LB: 1217.58193







xgb.grid.3 <- expand.grid(
      nrounds = 10, 
      max_depth = 12, 
      eta = .01, 
      gamma = c(2.5, 3, 3.5), 
      colsample_bytree = c(.75, .8, .85), 
      min_child_weight = c(3, 4, 5)
)



xgb.prep.2 <- train(x=xtrain[1:1000,],
                  y=ytrain[1:1000],
                  trControl = xgb.train.control,
                  tuneGrid = xgb.grid.3,
                  method = "xgbTree"
)


xgb.params.3 <- list(objective = "reg:linear",
                     booster = "gbtree",
                     nfold = 3,
                     max_depth = 12,
                     eta = .01,
                     gamma = 2.5,
                     colsample_bytree = 1,
                     min_child_weight = 1.3
)

xgb.3 <- xgboost(data = train, early_stopping_rounds = 5, nrounds = 500, params = xgb.params.3)

pred.test.xgb.3 <- predict(xgb.3, test)
head(pred.test.xgb.3)*mean.loss
#
#1799.8269 2309.3810 6700.6837 6183.5757  993.7541 2418.3592          LB: 1215.58867



pred.avg.1.2.3 <- (pred.test.xgb.1 + pred.test.xgb.2 + pred.test.xgb.3*3)*mean.loss/5
# 1819.602 2298.725 6368.988 6013.988 1026.732 2385.462                 LB: 1213.92844

sample$loss <- pred.avg.1.2.3
































write.csv(sample,'submission.11.3.5.csv',row.names = FALSE)











