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

down <- function(arg){as.numeric(unlist(as.data.frame(arg)))}

ytrain <- data.matrix(alldata[tr,1])
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain)
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))


xg_eval_mae <- function (yhat, dtrain) {
      y = getinfo(dtrain, "label")
      err= mae(exp(y),exp(yhat) )
      return (list(metric = "error", value = round(err, 3)))
}

train <- alldata[-c(fold11),]
valid <- alldata[fold11,]
test <- alldata[te,]

h2o.init(nthreads = -1)
h2o.removeAll()

train.hex <- as.h2o(train, "train.hex")
valid.hex <- as.h2o(valid, "valid.hex")
test.hex <- as.h2o(test, "test.hex")



features = 2:ncol(train.hex)
response = 1


dnn1<-h2o.deeplearning(x = features, 
                       y =response,
                       training_frame=train.hex,
                       validation_frame=valid.hex,
                       epochs=20, 
                       stopping_rounds=5,
                       overwrite_with_best_model=T,
                       activation="Rectifier",
                       distribution="huber",
                       hidden=c(100,100))


dnn2<-h2o.deeplearning(x = features,
                        y =response,
                        training_frame=train.hex,
                        validation_frame=valid.hex,
                        epochs=20,
                        stopping_rounds=5,
                        overwrite_with_best_model=T,
                        activation="Rectifier",
                        distribution="huber",
                        hidden=c(120, 100))


dnn3<-h2o.deeplearning(x = features,
                       y =response,
                       training_frame=train.hex,
                       validation_frame=valid.hex,
                       epochs=30, 
                       stopping_rounds=5,
                       overwrite_with_best_model=T,
                       activation="Rectifier",
                       distribution="huber",
                       hidden=c(120, 120))


dnn4<-h2o.deeplearning(x = features, 
                       y =response,
                       training_frame=train.hex,
                       validation_frame=valid.hex,
                       epochs=20, 
                       stopping_rounds=5,
                       overwrite_with_best_model=T,
                       activation="Rectifier",
                       distribution="huber",
                       hidden=c(80, 80, 80))

dnn5<-h2o.deeplearning(x = features, 
                       y =response,
                       training_frame=train.hex,
                       validation_frame=valid.hex,
                       epochs=50, 
                       stopping_rounds=5,
                       overwrite_with_best_model=T,
                       activation="Rectifier",
                       distribution="huber",
                       hidden=c(280, 280, 280))
# LB: 1165


pnn1 <- down(predict(dnn1, valid.hex[,-1]))
pnn2 <- down(predict(dnn2, valid.hex[,-1]))
pnn3 <- down(predict(dnn3, valid.hex[,-1]))
pnn4 <- down(predict(dnn4, valid.hex[,-1]))
pnn5 <- down(predict(dnn5, valid.hex[,-1]))















gbm.hyper_params = list( max_depth = c(4,6,8,12,16,20))

grid <- h2o.grid(
      hyper_params = gbm.hyper_params,
      search_criteria = list(strategy = "Cartesian"),
      algorithm="gbm",
      grid_id="gbm_grid1",
      x = features, 
      y = response, 
      training_frame = train.hex[1:1000,], 
      validation_frame = valid.hex[1:1000,],
      ntrees = 10000,              
      learn_rate = 0.05,           
      learn_rate_annealing = 0.99,                                               
      sample_rate = 0.8,                                                       
      col_sample_rate = 0.8, 
      seed = 7890,                                                             
      ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
      stopping_rounds = 5,
      stopping_tolerance = 1e-4,
      stopping_metric = "MSE"
)









gbm.params <- list(
      ntrees = 10000,
      seed = 0,
      stopping_rounds = 10)

h2o.gbm.1 <- function(..., params = gbm.params, max_depth = 4, rate = .2, rate_annealing = .9, sample_rate = .9, col_sample_rate = .8){ h2o.gbm.wrapper(..., params = params, rate_annealing = rate_annealing, sample_rate = sample_rate, col_sample_rate = col_sample_rate, max_depth = max_depth)}
h2o.gbm.2 <- function(..., params = gbm.params, max_depth = 8, rate = .1, rate_annealing = .95, sample_rate = .8, col_sample_rate = .8){h2o.gbm.wrapper(..., rate = rate, rate_annealing = rate_annealing, sample_rate = sample_rate, col_sample_rate = col_sample_rate, max_depth = max_depth)}
h2o.gbm.3 <- function(..., params = gbm.params, max_depth = 12, rate = .05, rate_annealing = .98, sample_rate = .8, col_sample_rate = .8){h2o.gbm.wrapper(..., rate = rate, rate_annealing = rate_annealing, sample_rate = sample_rate, col_sample_rate = col_sample_rate, max_depth = max_depth)}
h2o.gbm.4 <- function(..., params = gbm.params, max_depth = 10, rate = .01, rate_annealing = .99, sample_rate = .8, col_sample_rate = .8){h2o.gbm.wrapper(..., rate = rate, rate_annealing = rate_annealing, sample_rate = sample_rate, col_sample_rate = col_sample_rate, max_depth = max_depth)}
h2o.gbm.5 <- function(..., params = gbm.params, max_depth = 8, rate = .005, rate_annealing = .99, sample_rate = .8, col_sample_rate = .8){h2o.gbm.wrapper(..., rate = rate, rate_annealing = rate_annealing, sample_rate = sample_rate, col_sample_rate = col_sample_rate, max_depth = max_depth)}

dnn.params <- list(
      overwrite_with_best_model = TRUE,
      stopping_rounds = 10,
      stopping_metric = MSE,
      distribution = "huber"
)

h2o.dnn.1 <- function(..., params = dnn.params, hidden = c(30, 150, 50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.dnn.2 <- function(..., hidden = dnn.params, c(200,200,200), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.dnn.3 <- function(..., hidden = dnn.params, c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.dnn.4 <- function(..., hidden = c(500,200), activation = "Rectifier", epochs = 30, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation,  seed = seed)
h2o.dnn.5 <- function(..., hidden = c(200,100,50), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.dnn.6 <- function(..., hidden = c(50,100,50), activation = "Rectifier", epochs = 30, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)



learner <- c("h2o.glm.wrapper", "h2o.gbm.1", "h2o.gbm.2", "h2o.gbm.3", "h2o.gbm.4", "h2o.gbm.5") #, "h2o.deeplearning.wrapper")
metalearner <- "h2o.glm.wrapper"

ens.1 <- h2o.ensemble(x = features, y = response, 
                      training_frame = train.hex, 
                      validation_frame = valid.hex,
                      family = "gaussian", 
                      learner = learner, 
                      metalearner = metalearner)

pred.train.ens.1 <- down(predict(ens.1, train.hex)$pred)
pred.valid.ens.1 <- down(predict(ens.1, valid.hex)$pred)
pred.test.ens.1 <- down(predict(ens.1, test.hex)$pred)
train.df <- pred.train.ens.1$basepred
valid.df <- pred.valid.ens.1$basepred
test.df <- pred.test.ens.1$basepred


t <- data.frame(p1 = pnn1, p2 = pnn2, p3 = pnn3, p4 = pnn4, p5 = pnn5, 
                p6 = down(test.df[,1]), p7 = down(test.df[,2]),
                p8 = down(test.df[,3]), p9 = pred.xgb, p10 = pred.gbdt)










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


xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])
xtrain.cv <- xgb.DMatrix(data.matrix(alldata[1:150000,-1]), label = ytrain[1:150000])
watch.cv <- xgb.DMatrix(data.matrix(alldata[150001:188318,-1]), label = ytrain[150001:188318])


xgb.params <- list(
      seed = 0,
      objective = "reg:linear",
      eta = 0.005,
      max_depth = 12,
      alpha = 1,
      gamma = 2,
      colsample_bytree = .55,
      min_child_weight = 1,
      subsample = .8,
      base_score = 7.76
)
xgb10 <- xgboost(data = xtrain,
                   params = xgb.params,
                  feval = xg_eval_mae,
                  maximize = FALSE,
                  nrounds = 6000, 
                  print_every_n = 10
                  )

# LB: 1116















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

pred.train.ens <- down(predict(ens, px.hex)$pred)))
pred.test.ens <- down(predict(ens, pt.hex)$pred)))

head(pred.test.ens)
sample$loss <- exp(pred.test.ens)

# write.csv(sample, "sample.sub.11.17.1.csv", row.names = FALSE)   LB: 1124.59


