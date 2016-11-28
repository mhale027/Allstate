library(data.table)
library(xgboost)
library(caret)
library(Matrix)
library(h2o)
library(h2oEnsemble)
library(Metrics)


set.seed(7890)
setwd("~/projects/kaggle/allstate")
load("temp.nzv.add.200.RData")
# source("features.R")
# load("template.R")

# setwd("/users/matt/documents/projects/kaggle/allstate")
training <- data.frame(fread("train.csv", header = TRUE))
test <- data.frame(fread("test.csv", header = TRUE))
sample <- data.frame(fread("sample_submission.csv", header = TRUE))
training$loss <- log(training$loss)
length.train <- nrow(training)
legnth.test <- nrow(test)
training.id <- training$id
test.id <- test$id
training.loss <- training$loss
mean.loss <- mean(training$loss)
training.length <- nrow(training)
test.length <- nrow(test)
tr <- 1:training.length
te <- (1+training.length):(training.length + test.length)
rm(training, test)

load("template.RData")

fold1 <- 1:18000
fold2 <- 18001:36000
fold3 <- 36001:54000
fold4 <- 54001:72000
fold5 <- 72001:90000
fold6 <- 90001:108000
fold7 <- 108001: 126000
fold8 <- 126001:144000
fold9 <- 144001: 162000
fold10 <- 162001:180000
fold11 <- 180001:188318



f1 <- tr[-c(fold1)]
f2 <- tr[-c(fold2)]
f3 <- tr[-c(fold3)]
f4 <- tr[-c(fold4)]
f5 <- tr[-c(fold5)]
f6 <- tr[-c(fold6)]
f7 <- tr[-c(fold7)]
f8 <- tr[-c(fold8)]
f9 <- tr[-c(fold9)]
f10 <- tr[-c(fold10)]


ytrain <- data.matrix(alldata[tr,1])
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain)
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))


xg_eval_mae <- function (yhat, dtrain) {
      y = getinfo(dtrain, "label")
      err= mae(exp(y),exp(yhat) )
      return (list(metric = "error", value = round(err, 3)))
}




xtrain.cv <- xgb.DMatrix(data.matrix(alldata[f1,-1]), label = ytrain[f1])
watch.cv <- xgb.DMatrix(data.matrix(alldata[fold1,-1]), label = ytrain[fold1])

nrounds = 3500

xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.05,
      max_depth = 4,
      alpha = 1,
      gamma = 2,
      colsample_bytree = 1,
      min_child_weight = 1,
      subsample = 1,
      base_score = 7.76
)

xgb1 <- xgb.train(xtrain.cv, params = xgb.params,
                  watchlist = list(val1 = watch.cv),
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = nrounds,
                  early_stopping_rounds = 20,
                  print_every_n = 15)
# error: adjust + 200, 101 vars  1136.98 nrounds 2500
# error: adjust + 200, nzv-110 vars nfold4 1137 ; 1534 rounds nfold1:1138.27 1610rounds
# error: no adjust   , 140 vars, nofold; 1139 at 1500 rounds
# error: adjust + 200, 140 vars, nofold; 1136.0 @ 2464 rounds



xtrain.cv <- xgb.DMatrix(data.matrix(alldata[f2,-1]), label = ytrain[f2])
watch.cv <- xgb.DMatrix(data.matrix(alldata[fold2,-1]), label = ytrain[fold2])

xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.1,
      max_depth = 4,
      alpha = 0,
      gamma = 1,
      colsample_bytree = .7,
      min_child_weight = 1,
      subsample = .8,
      base_score = 7.76
)
xgb2 <- xgb.train(xtrain.cv, params = xgb.params,
                  watchlist = list(val1 = watch.cv),
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = nrounds, 
                  early_stopping_rounds = 20,
                  print_every_n = 15)
# error 1135  
# adjust + 200, 140 vars, nofold; 1134.0 @ 2457 rounds


xtrain.cv <- xgb.DMatrix(data.matrix(alldata[f3,-1]), label = ytrain[f3])
watch.cv <- xgb.DMatrix(data.matrix(alldata[fold3,-1]), label = ytrain[fold3])

xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.1,
      max_depth = 12,
      alpha = 1,
      gamma = 2,
      colsample_bytree = .5,
      min_child_weight = 1,
      subsample = .8,
      base_score = 7.76
)
xgb3 <- xgb.train(xtrain.cv, params = xgb.params,
                  watchlist = list(val1 = watch.cv),
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = nrounds, 
                  early_stopping_rounds = 20,
                  print_every_n = 15)
# adjust + 200, 140 vars, nofold; 1141.99 @ 2674 rounds


xtrain.cv <- xgb.DMatrix(data.matrix(alldata[f4,-1]), label = ytrain[f4])
watch.cv <- xgb.DMatrix(data.matrix(alldata[fold4,-1]), label = ytrain[fold4])

xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.05,
      max_depth = 8,
      alpha = 1,
      gamma = 2,
      colsample_bytree = .8,
      min_child_weight = 1,
      subsample = .9,
      base_score = 7.76
)
xgb4 <- xgb.train(xtrain.cv, params = xgb.params,
                  watchlist = list(val1 = watch.cv),
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = nrounds,
                  early_stopping_rounds = 20,
                  print_every_n = 15)
# adjust + 200, 140 vars, nofold; 1141.326 @ 2226 rounds

xtrain.cv <- xgb.DMatrix(data.matrix(alldata[f5,-1]), label = ytrain[f5])
watch.cv <- xgb.DMatrix(data.matrix(alldata[fold5,-1]), label = ytrain[fold5])


xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.05,
      max_depth = 14,
      alpha = 1,
      gamma = 3,
      colsample_bytree = .6,
      min_child_weight = 1,
      subsample = .8,
      base_score = 7.76
)
xgb5 <- xgb.train(xtrain.cv, params = xgb.params,
                  watchlist = list(val1 = watch.cv),
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = nrounds, 
                  early_stopping_rounds = 20,
                  print_every_n = 15)


# adjust + 200, 140 vars, nofold; 1125.768 @ 2727 rounds

# error 1136



gc()
xtrain.cv <- xgb.DMatrix(data.matrix(alldata[f6,-1]), label = ytrain[f6])
watch.cv <- xgb.DMatrix(data.matrix(alldata[fold6,-1]), label = ytrain[fold6])


xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.03,
      max_depth = 12,
      alpha = 0,
      gamma = 0,
      colsample_bytree = .9,
      min_child_weight = 1,
      subsample = .8,
      base_score = 7.76
)
xgb6 <- xgb.train(xtrain.cv, params = xgb.params,
                  watchlist = list(val1 = watch.cv),
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = nrounds, 
                  early_stopping_rounds = 20,
                  print_every_n = 15)



xtrain.cv <- xgb.DMatrix(data.matrix(alldata[f7,-1]), label = ytrain[f7])
watch.cv <- xgb.DMatrix(data.matrix(alldata[fold7,-1]), label = ytrain[fold7])


xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.03,
      max_depth = 8,
      alpha = 1,
      gamma = 1,
      colsample_bytree = .5,
      min_child_weight = 1,
      subsample = .8,
      base_score = 7.76
)
xgb7 <- xgb.train(xtrain.cv, params = xgb.params,
                  watchlist = list(val1 = watch.cv),
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = nrounds, 
                  early_stopping_rounds = 20,
                  print_every_n = 15)


xtrain.cv <- xgb.DMatrix(data.matrix(alldata[f8,-1]), label = ytrain[f8])
watch.cv <- xgb.DMatrix(data.matrix(alldata[fold8,-1]), label = ytrain[fold8])


xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.01,
      max_depth = 10,
      alpha = 1,
      gamma = 5,
      colsample_bytree = .8,
      min_child_weight = 1,
      subsample = .8,
      base_score = 7.76
)
xgb8 <- xgb.train(xtrain.cv, params = xgb.params,
                  watchlist = list(val1 = watch.cv),
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = nrounds, 
                  early_stopping_rounds = 20,
                  print_every_n = 15)


xtrain.cv <- xgb.DMatrix(data.matrix(alldata[f9,-1]), label = ytrain[f9])
watch.cv <- xgb.DMatrix(data.matrix(alldata[fold9,-1]), label = ytrain[fold9])


xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.01,
      max_depth = 12,
      alpha = 1,
      gamma = 2,
      colsample_bytree = .5,
      min_child_weight = 1,
      subsample = .8,
      base_score = 7.76
)
xgb9 <- xgb.train(xtrain.cv, params = xgb.params,
                  watchlist = list(val1 = watch.cv),
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = nrounds, 
                  early_stopping_rounds = 20,
                  print_every_n = 15)


xtrain.cv <- xgb.DMatrix(data.matrix(alldata[f10,-1]), label = ytrain[f10])
watch.cv <- xgb.DMatrix(data.matrix(alldata[fold10,-1]), label = ytrain[fold10])


xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.005,
      max_depth = 10,
      alpha = 2,
      gamma = 3,
      colsample_bytree = .5,
      min_child_weight = 1,
      subsample = .7,
      base_score = 7.76
)
xgb10 <- xgb.train(xtrain.cv, params = xgb.params,
                  watchlist = list(val1 = watch.cv),
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = 6000, 
                  early_stopping_rounds = 20,
                  print_every_n = 15)

















pval1 <- predict(xgb1, watch1)
pval2 <- predict(xgb2, watch2)
pval3 <- predict(xgb3, watch3)
pval4 <- predict(xgb4, watch4)
pval5 <- predict(xgb5, watch5)
oob.train <- c(pval1, pval2, pval3, pval4, pval5)


px.1 <- predict(xgb1, xtrain)
px.2 <- predict(xgb2, xtrain)
px.3 <- predict(xgb3, xtrain)
px.4 <- predict(xgb4, xtrain)
px.5 <- predict(xgb5, xtrain)

ptrain <- data.frame(loss = alldata$loss[tr], p1 = px.1, p2 = px.2, p3 = px.3, p4 = px.4, p5 = px.5)

# p1: 1141
# p2: 1136
# p3: 1151
# p4: 1126
# p5: 1138

pt.1 <- predict(xgb1, xtest)
pt.2 <- predict(xgb2, xtest)
pt.3 <- predict(xgb3, xtest)
pt.4 <- predict(xgb4, xtest)
pt.5 <- predict(xgb5, xtest) # train mae: 1042; val mae: 1125.768, LB: 1120.27957
ptest <- data.frame(p1 = pt.1, p2 = pt.2, p3 = pt.3, p4 = pt.4, p5 = pt.5)
pt.mean <- (pt.1 + pt.2 + pt.3 + pt.4 + pt.5)/5 # LB: 

p.all <- cbind(data.frame(loss = alldata$loss), rbind(ptrain, ptest))

pred.avg <- as.numeric(apply(rbind(ptrain,ptest), 1, mean))
sample$loss <- exp(pred.avg) - 200
write.csv(sample, "sample.sub.11.18.2.1.csv", row.names = FALSE)

h2o.init(nthreads = -1)
h2o.removeAll() 

pred.hex <- as.h2o(p.all)

px.hex <- pred.hex[tr,]
pt.hex <- pred.hex[te,]

p.sets <- h2o.splitFrame(px.hex, .95)

px.valid.hex <- h2o.assign(p.sets[[2]], "px.valid.hex")
px.hex <- h2o.assign(p.sets[[1]], "px.hex")

response <- 1
features <- 2:5

learner <- c("h2o.glm.wrapper", "h2o.gbm.wrapper", "h2o.randomForest.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "h2o.glm.wrapper"

ens <- h2o.ensemble(x = features, y = response,
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

head(pred.test.ens)
sample$loss <- exp(pred.test.ens)

# write.csv(sample, "sample.sub.11.17.1.csv", row.names = FALSE)   LB: 1124.59


