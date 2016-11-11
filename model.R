library(data.table)
library(randomForest)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(Metrics)
library(mlbench)
library(h2o)
library(h2oEnsemble)


setwd("~/Projects/Kaggle/Allstate")
set.seed(7890)

load("alldata.final.RData")
load("features.RData")

training <-data.frame(fread("train.csv", header = TRUE))
test <- data.frame(fread("test.csv", header = TRUE))
sample <- data.frame(fread("sample_submission.csv", header = TRUE))

lengths <- features[[2]]
training.length <- lengths[1]
test.length <- lengths[2]

tr <- 1:training.length
te <- (training.length+1):(training.length + test.length)
      
      
ids <- features[[3]]
training.id <- ids[tr]
test.id <- ids[te]

losses <- features[[4]]
training.loss <- losses[tr]
test.loss <- losses[te]

mean.loss <- mean(training.loss)

load("inTrain.RData")

rm(features, lengths, ids, losses, lns, thous)

gc()


# 
# 
# rf.tr <- randomForest(y=training.loss[1:1000], x=training.set[1:1000,], importance = TRUE, ntree = 200)
# imp.rf <- varImp(rf.tr)
# imp.rf$var <- rownames(imp.rf)
# imp.rf <- c(arrange(imp.rf, desc(Overall))$var)
# save(rf.tr, file = "rf.tr.00.RData")
# # load("rf.tr.RData")
# 
# 
# rf.vars <- unique(c(imp.rf[1:20]))
# 
# vars <- c()
# 
# for (i in 1:length(rf.vars)){
#  subs <- names(training.set)[grep(gsub("_.*", "_", rf.vars[i]), names(training.set))]
#  rf.vars <- unique(c(rf.vars, subs))
# }
# 
# gc()
# 
# training.rf.1 <- training.set[,rf.vars]
# test.rf.1 <- test.set[,rf.vars]
# 
# save(training.rf.1, file = "training.rf.1.00.RData")
# save(test.rf.1, file = "test.rf.1.00.RData")
# 
# # load("training.rf.1.00.RData")
# # load("test.rf.1.00.RData")


gc()

training.guess.1 <- rep(0, training.length)
for (i in tr) { training.guess.1[i] <- ifelse(training.loss > mean.loss, 1, 0) }


load("alldata.nzv.RData")


nnet.1 <- nnet(x=data.matrix(alldata[tr,]), y=data.matrix(training.loss/mean.loss), size = 5, linout = TRUE)
pred.nnet.1 <- predict(nnet.1, alldata[tr,], type = "raw")
pred.nnet.1 <- pred.nnet.1 * mean.loss

pred.test.nnet.1 <- predict(nnet.1, alldata[te,], type = "raw")
pred.test.nnet.1 <- pred.test.nnet.1 * mean.loss





# 
# 
# 
# inp <- mx.symbol.Variable('data')
# l1 <- mx.symbol.FullyConnected(inp, name = "l1", num.hidden = 100)
# a1 <- mx.symbol.Activation(l1, name = "a1", act_type = 'relu')
# d1 <- mx.symbol.Dropout(a1, name = 'd1', p = 0.4)
# l2 <- mx.symbol.FullyConnected(d1, name = "l2", num.hidden = 50)
# a2 <- mx.symbol.Activation(l2, name = "a2", act_type = 'relu')
# d2 <- mx.symbol.Dropout(a2, name = 'd2', p = 0.2)
# l3 <- mx.symbol.FullyConnected(d2, name = "l3", num.hidden = 1)
# outp <- mx.symbol.MAERegressionOutput(l3, name = "outp")
# 
# 
# mx.1 <- mx.model.FeedForward.create(outp, 
#                                X = as.array(t(alldata[tr[inTrain],])), 
#                                y = as.array(training.loss/mean.loss),
#                                eval.data =
#                                      list(data = as.array(t(alldata[tr[-inTrain],])),
#                                           label = as.array(alldata[test_obs,])),
#                                array.layout = 'colmajor',
#                                # eval.metric=mx.metric.mae,
#                                learning.rate = .01,
#                                # momentum = params$momentum,
#                                # wd = params$wd,
#                                # array.batch.size = params$batch.size,
#                                num.round = 10)

mx.1 <- mx.mlp(data.matrix(alldata[tr,]), as.array(training.loss/mean.loss), 
               hidden_node=10, 
               out_node=1, 
               out_activation="rmse",
               num.round=20, 
               array.batch.size=15, 
               learning.rate=0.07, 
               momentum=0.9, 
               eval.metric=mx.metric.accuracy,
               array.layout = "rowmajor",
               ctx = mx.cpu()
)










controls.1 <- trainControl(method = "repeadedcv",
                           number = 5,
                           repeat = 3)

mtry.1 = floor(sqrt(ncol(training.rf.1)))
ntrees.1 = 200


rf.1 <- randomForest(y=training.loss, x=training.rf.1, 
                     trControl = controls.1, 
                     mtry = mtry.1,
                     ntree = ntrees.1)

save(rf.1, file = "rf.1.00.RData")
# load("rf.1.00.RData")


pred.rf.1.train <- predict(rf.1, training.rf.1)
pred.rf.1.test <- predict(rf.1, test.rf.1)

rm(training.rf.1, test.rf.1)






gc()




labels <- data.matrix(training.loss)
xtrain.xgb.1 <- xgb.DMatrix(data = data.matrix(training.set), label = labels)

params.xgb.1 <- list(
      booster = "gblinear",
      objective = "reg:linear", 
      eta = 0.05, 
      subsample = .7, 
      colsample_bytree = 0.7, 
      min_child_weight = 1,
      base_score = 7.69,
      num_parallel_tree = 1
)

xg_eval_mae <- function (yhat, data) {
      labels = getinfo(data, "label")
      err= as.numeric(mean(abs(labels - yhat)))
      return (list(metric = "error", value = err))
}

gc()

res <- xgb.cv(data = xtrain.xgb.1,
                 params = params.xgb.1,
                 early_stopping_rounds = 10,
                 nfold = 4,
                 nrounds = 750,
                 feval = xg_eval_mae,
                 maximize = FALSE
)

gc()

nrounds.xgb.1 <- res$best_iteration
nfold.xgb.1 <- 4

labels <- data.matrix(training.loss)
xtrain.xgb.1 <- xgb.DMatrix(data.matrix(training.set), label = labels)


gc()


xgb.1 <- xgboost(data = xtrain.xgb.1, 
                 params = params.xgb.1,
                 nrounds = nrounds.xgb.1, 
                 nfold = nfold)


gc()

y.xgb.1 <- xgb.DMatrix(data.matrix(test.set))

pred.xgb.1.train <- predict(xgb.1, xtrain.xgb.1)
pred.xgb.1.test <- predict(xgb.1, y.xgb.1)

rm(xtrain.xgb.1, y.xgb.1)

gc()











