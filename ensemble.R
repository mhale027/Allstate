library(data.table)
library(randomForest)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(Metrics)
library(h2o)
library(h2oEnsemble)


setwd("/users/matt/documents/projects/kaggle/allstate")
set.seed(7890)
load("template.RData")
source("features.R")


h2o.init(nthreads = -1)  
h2o.removeAll() 

train <- as.h2o(alldata[tr,1:140])
test.hex <- as.h2o(alldata[te,1:140])

sets <- h2o.splitFrame(train, .6, seed = 1)

train.hex <- h2o.assign(sets[[1]], "train.hex")
valid.hex <- h2o.assign(sets[[2]], "valid.hex")


response <- 1
features <- 2:140

learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "h2o.glm.wrapper"

ens.1 <- h2o.ensemble(x = features, y = response, 
                    training_frame = train.hex, 
                    validation_frame = valid.hex,
                    family = "gaussian", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5))

pred.train.ens.1 <- as.numeric(unlist(as.data.frame(predict(ens.1, train.hex)$pred)))
pred.valid.ens.1 <- as.numeric(unlist(as.data.frame(predict(ens.1, valid.hex)$pred)))
pred.test.ens.1 <- as.numeric(unlist(as.data.frame(predict(ens.1, test.hex)$pred)))

g1 <- data.frame(g1 = c(pred.train.ens.1, pred.valid.ens.1, pred.test.ens.1))
sample$loss <- pred.test.ens.1
#write.csv(sample, "sample.submission.11.12.3.csv", row.names = FALSE)
#         LB: 1201.90208
#save(ens.1, file = "ens.1.RData")





# train on all data where test losses are the esimate from predict 


alldata[te,1] <- pred.test.ens.1

train <- as.h2o(alldata)

sets <- h2o.splitFrame(train, .8, seed = 1)

train.hex <- h2o.assign(sets[[1]], "train.hex")
valid.hex <- h2o.assign(sets[[2]], "valid.hex")

#rm(train, sets)

ens.1.1 <- h2o.ensemble(x = features, y = response, 
                    training_frame = train.hex, 
                    validation_frame = valid.hex,
                    family = "gaussian", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5)
)

train.hex <- as.h2o(alldata[tr,])

pred.ens.1.1 <- as.numeric(unlist(as.data.frame(predict(ens.1.1, train)$pred)))


#write.csv(sample, "sample.submission.11.12.3.csv", row.names = FALSE)
#     LB: 1188.13732
#     LB: 

alldata <- cbind(alldata, pred.ens.1.1)

tr.1 <- 1:130000
tr.2 <- 130001:188318
xgb.train <- alldata[tr,]
xgb.test <- alldata[te,]

ytrain <- data.matrix(alldata[tr.1, 1])
xtrain <- xgb.DMatrix(data.matrix(xgb.train[tr.1,-1]))
yvalid <- data.matrix(alldata[tr.2, 1])
xvalid <- xgb.DMatrix(data.matrix(xgb.train[tr.2,-1]))
xtest <- xgb.DMatrix(data.matrix(xgb.test[,-1]), label = xgb.test[,1 ])

xgb.grid.f1.1 <- expand.grid(
    nrounds = 10,
    eta = .05,
    max_depth = c(8, 10, 14),
    gamma = c(0, 3),
    colsample_bytree = c(0.7, 0.75, .8),
    min_child_weight = 1,
    subsample = c(.8, .9)
    )



xgb.train.control.f1.1 <- trainControl(
    method = "cv",
    number = 4,
    verboseIter = TRUE,
    returnData = FALSE,
    returnResamp = "all",
    allowParallel = TRUE
)


xgb.prep.f1.1 <- train(x = alldata[1:1000,-1],
                       y=alldata[1:1000,1],
                       trControl = xgb.train.control.f1.1,
                       tuneGrid = xgb.grid.f1.1,
                       method = "xgbTree",
                       metric = "RMSE"
)



xgb.f1.1.max_depth <- xgb.prep.f1.1$bestTune$max_depth
xgb.f1.1.gamma <- xgb.prep.f1.1$bestTune$gamma
xgb.f1.1.colsample_bytree <- xgb.prep.f1.1$bestTune$colsample_bytree
xgb.f1.1.min_child_weight <- xgb.prep.f1.1$bestTune$min_child_weight

xgb.f1.1.params <- list(
    objective = "reg:linear",
    num_class = 2,
    eta = 0.05,
    max_depth = xgb.f1.1.max_depth, 
    gamma = xgb.f1.1.gamma,
    colsample_bytree = xgb.f1.1.colsample_bytree,
    min_child_weight = xgb.f1.1.min_child_weight,
    metric = "merror",
    feval = "mae"
)

res.f1.1 <- xgb.cv(data = xtrain,
                   label = ytrain,
                   nrounds = 500,
                   nfold = 4,
                   params = xgb.f1.1.params,
                   early_stopping_rounds = 10,
                   print_every_n = 10
)

nrounds.f1.1 <- res.f1.1$best_iteration     

xgb.f1.1 <- xgb.train(data = xtrain, 
                    params = xgb.f1.1.params,
                    watchlist = xval,
                    nfold = 4, 
                    nrounds = nrounds.f1.1, 
                    early_stopping_rounds = 10)

pred.train.xgb.f1.1 <- predict(xgb.f1.1, xtrain)
pred.test.xgb.f1.1 <- predict(xgb.f1.1, xtest)
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              