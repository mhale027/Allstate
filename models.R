library(data.table)
library(xgboost)
library(caret)
library(Matrix)
library(h2o)
library(h2oEnsemble)
library(Metrics)


set.seed(7890)
setwd("~/projects/kaggle/allstate")
load("template.RData")
source("features.R")

ytrain <- data.matrix(alldata[tr,1])
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain)
xtrain1 <- xgb.DMatrix(data.matrix(alldata[f1,-1]), label = ytrain[f1])
xtrain2 <- xgb.DMatrix(data.matrix(alldata[f2,-1]), label = ytrain[f2])
xtrain3 <- xgb.DMatrix(data.matrix(alldata[f3,-1]), label = ytrain[f3])
xtrain4 <- xgb.DMatrix(data.matrix(alldata[f4,-1]), label = ytrain[f4])
xtrain5 <- xgb.DMatrix(data.matrix(alldata[f5,-1]), label = ytrain[f5])
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))
watch1 <- xgb.DMatrix(data.matrix(alldata[fold1,-1]), label = ytrain[fold1])
watch2 <- xgb.DMatrix(data.matrix(alldata[fold2,-1]), label = ytrain[fold2])
watch3 <- xgb.DMatrix(data.matrix(alldata[fold3,-1]), label = ytrain[fold3])
watch4 <- xgb.DMatrix(data.matrix(alldata[fold4,-1]), label = ytrain[fold4])
watch5 <- xgb.DMatrix(data.matrix(alldata[fold5,-1]), label = ytrain[fold5])

xgb.grid <- expand.grid(
      nrounds = 20,
      eta = .07,
      max_depth = c(9, 10, 11,12),
      gamma = c(3,4,5, 5.5, 6),
      colsample_bytree = c(.45, .5, .55, .6),
      min_child_weight = c(3.25, 3.5, 3.75, 4, 5)
)

xgb.train.control <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      allowParallel = TRUE
)


xgb.prep <- train(x = alldata[1:1000,-1],
                  y=alldata[1:1000,1],
                  trControl = xgb.train.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  metric = "RMSE"
)
# Fitting nrounds = 20, max_depth = 13, eta = 0.2, gamma = 5.5, colsample_bytree = 0.55, min_child_weight = 5 
# Fitting nrounds = 20, max_depth = 11, eta = 0.07, gamma = 5.5, colsample_bytree = 0.55, min_child_weight = 3.75
# Fitting nrounds = 20, max_depth = 11, eta = 0.07, gamma = 4, colsample_bytree = 0.55, min_child_weight = 5 <- using this


xgb.max_depth <- xgb.prep$bestTune$max_depth
xgb.gamma <- xgb.prep$bestTune$gamma
xgb.colsample_bytree <- xgb.prep$bestTune$colsample_bytree
xgb.min_child_weight <- xgb.prep$bestTune$min_child_weight

xgb.params <- list(
      objective = "reg:linear",
      eta = 0.07,
      max_depth = xgb.max_depth,
      gamma = xgb.gamma,
      colsample_bytree = xgb.colsample_bytree,
      min_child_weight = xgb.min_child_weight,
      metric = "rmse"
)

res <- xgb.cv(data = xtrain,
                   nrounds = 1500,
                   nfold = 4,
                   params = xgb.params,
                   early_stopping_rounds = 5,
                   print_every_n = 10
)

nrounds <- res$best_iteration     


xgb1 <- xgb.train(xtrain1, params = xgb.params,
                  watchlist = list(val1 = watch1),
                    nfold = 4, 
                    nrounds = nrounds, 
                    early_stopping_rounds = 10)


xgb2 <- xgb.train(xtrain2, params = xgb.params,
                  watchlist = list(val1 = watch2),
                  nfold = 4, 
                  nrounds = nrounds, 
                  early_stopping_rounds = 10)


xgb3 <- xgb.train(xtrain3, params = xgb.params,
                  watchlist = list(val1 = watch3),
                  nfold = 4, 
                  nrounds = nrounds, 
                  early_stopping_rounds = 10)


xgb4 <- xgb.train(xtrain4, params = xgb.params,
                  watchlist = list(val1 = watch4),
                  nfold = 4, 
                  nrounds = nrounds, 
                  early_stopping_rounds = 10)


xgb5 <- xgb.train(xtrain5, params = xgb.params,
                  watchlist = list(val1 = watch5),
                  nfold = 4, 
                  nrounds = nrounds, 
                  early_stopping_rounds = 10)


px.1 <- predict(xgb1, xtrain)
px.2 <- predict(xgb2, xtrain)
px.3 <- predict(xgb3, xtrain)
px.4 <- predict(xgb4, xtrain)
px.5 <- predict(xgb5, xtrain)
ptrain <- data.frame(loss = training.loss, p1 = px.1, p2 = px.2, p3 = px.3, p4 = px.4, p5 = px.5)

pt.1 <- predict(xgb1, xtest)
pt.2 <- predict(xgb2, xtest)
pt.3 <- predict(xgb3, xtest)
pt.4 <- predict(xgb4, xtest)
pt.5 <- predict(xgb5, xtest)
ptest <- data.frame(loss = rep(NA, length(te)), p1 = pt.1, p2 = pt.2, p3 = pt.3, p4 = pt.4, p5 = pt.5)

p.all <- rbind(ptrain, ptest)

h2o.init(nthreads = -1)
h2o.removeAll() 

alldata <- as.h2o(rbind(ptrain, ptest))
pred.hex <- as.h2o(p.all)


train.h <- as.h2o(alldata[tr,])
test.hex <- as.h2o(alldata[te,])

px.hex <- pred.hex[tr,]
pt.hex <- pred.hex[te,]

sets <- h2o.splitFrame(train.h, .95)
p.sets <- h2o.splitFrame(px.hex, .95)

valid.hex <- h2o.assign(sets[[2]], "valid.hex")
train.hex <- h2o.assign(sets[[1]], "train.hex")
px.valid.hex <- h2o.assign(p.sets[[2]], "px.valid.hex")
px.hex <- h2o.assign(p.sets[[1]], "px.hex")

response <- 1
features <- 2:140

learner <- c("h2o.glm.wrapper", "h2o.gbm.wrapper", "h2o.randomForest.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "h2o.glm.wrapper"

ens <- h2o.ensemble(x = 2:5, y = 1, 
                    model_id = "ens",
                    training_frame = px.hex, 
                    validation_frame = px.valid.hex,
                    family = "gaussian", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5)
)

pred.train.ens <- as.numeric(unlist(as.data.frame(predict(ens, px.hex)$pred)))
pred.test.ens <- as.numeric(unlist(as.data.frame(predict(ens, pt.hex)$pred)))
