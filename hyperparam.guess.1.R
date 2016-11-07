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
load("features.RData")
load("data.RData")


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

mean.loss <- mean(training.loss)


guess.1 <- rep(0, nrow(alldata))

for (i in 1:nrow(alldata)) {
      guess.1[i] <- ifelse(training.loss[i] > mean.loss, 1, 0)
}

alldata$loss <- as.factor(guess.1)



x.vars <- names(alldata)[-1]
response <- names(alldata)[1]


xtrain <- alldata[tr,-1]
xtest <- alldata[te,-1]
ytrain <- alldata[tr,1]


train <- xgb.DMatrix(data.matrix(xtrain), label = data.matrix(ytrain))
test <- xgb.DMatrix(data.matrix(xtest))





###################GRID SEARCH###################

# xgb <- xgboost(data = train, nrounds = 1)
# 
# base.margin <- predict(xgb.1, train, outputmargin = TRUE)
# 
# setinfo(xtrain, "base_margin", base.margin) 




xgb.grid <- expand.grid(
      nrounds = 10, 
      eta = 0.01, 
      max_depth = c(10, 12, 14), 
      gamma = c(0, 3, 5), 
      colsample_bytree = c(0.7, 0.85, 1.0), 
      min_child_weight = c(0, 1, 2)
)

xgb.train.control <- trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      allowParallel = TRUE
)


xgb.prep <- train(x=alldata[1:1000,-1],
                  y=alldata[1:1000,1],
                  trControl = xgb.train.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  metric = "RMSE"
)



xgb.1.max_depth <- xgb.prep$bestTune$max_depth
xgb.1.gamma <- xgb.prep$bestTune$gamma
xgb.1.colsample_bytree <- xgb.prep$bestTune$colsample_bytree
xgb.1.min_child_weight <- xgb.prep$bestTune$min_child_weight

xgb.1.params <- list(
      objective = "reg:linear",
      eta = 0.01,              
      max_depth = xgb.1.max_depth, 
      gamma = xgb.1.gamma,
      colsample_bytree = xgb.1.colsample_bytree,
      min_child_weight = xgb.1.min_child_weight,
      eval_metric = "rmse"      
)


xgb.1 <- xgboost(train, params = xgb.1.params, nfold = 4, nrounds = 200, early_stopping_rounds = 5)

pred.train.xgb.1 <- predict(xgb.1, train)
pred.test.xgb.1 <- predict(xgb.1, test)

head(pred.train.xgb.1)
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












load("training.rf.2.0.RData")
load("test.rf.2.0.RData")


training.guess.1 <- rep(0, length(training.loss))


for (i in 1:length(training.loss)) {
      training.guess.1[i] <- ifelse(training.loss[i] > mean.loss, 1, 0)
}

training.rf.2 <- cbind(data.frame(loss = as.factor(training.guess.1)), training.rf.2)

rf.1 <- ranger(loss~., 
                 data = training.rf.2, 
                 num.trees = 10,
                 mtry = 5, 
                 write.forest = TRUE,
                 verbose = TRUE
)

pred.rf.guess.1 <- predict(rf.1, training.rf.2)
head(pred.rf.guess.1$predictions)
gc()



xgb.guess.1 <- xgboost(data = xgb.DMatrix(data.matrix(alldata[tr,-1]), label = data.matrix(training.guess.1)),
                       nrounds = 100,
                       early_stopping_rounds = 5)















write.csv(sample,'submission.11.3.5.csv',row.names = FALSE)











