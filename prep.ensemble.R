library(data.table)
library(randomForest)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(Metrics)


setwd("~/Projects/Kaggle/Allstate")
set.seed(7890)
source("sparse.R")

training <-data.frame(fread("train.csv", header = TRUE))
test <- data.frame(fread("test.csv", header = TRUE))
sample <- data.frame(fread("sample_submission.csv", header = TRUE))

test$loss <- 0



 
inTrain <- createDataPartition(training$loss, p=.7, list = FALSE)
training <- training[inTrain,]
validate <- training[-inTrain,]


#create new vectors with the loss amounts and ids so they can be removed from the training set
training.id <- training$id
validate.id <- validate$id
test.id <- test$id
ids <- c(training.id, validate.id, test.id)


training.t <- round(training$loss/1000)
validate.t <- round(validate$loss/1000)
test.t <- round(test$loss/1000)
thous <- c(training.t, validate.t, test.t)




training.loss <- log(training$loss)
validate.loss <- log(validate$loss)
test.loss <- log(test$loss)
losses <- c(training.loss, validate.loss, test.loss)

training.ln <- round(training.loss)
validate.ln <- round(validate.loss)
test.ln <- round(test.loss)
lns <- c(training.ln, validate.ln, test.ln)



training.l <- nrow(training)
validate.l <- nrow(validate)
test.l <- nrow(test)
lengths <- c(training.l, validate.l, test.l)

training <- select(training, -id, -loss)
validate <- select(validate, -id, -loss)
test <- select(test, -id, -loss)

gc()

 # alldata <- rbind(training, validate, test)
 # 
 # 
 # 
 # nzv <- nearZeroVar(alldata)
 # 
 # alldata <- alldata[,-nzv]
 # 
 # clean.set <- function(input) {
 # 
 #       for (i in 1:length(grep("cat", names(input)))) {
 #             new.col <- input[,i]
 #             choices <- unique(new.col)
 #             for (k in 1:length(choices)) {
 #                   new.col[new.col == choices[k]] <- k
 #             }
 #             input[,i] <- as.factor(new.col)
 #       }
 #       input <- data.frame(input)
 # }
 # 
 # gc()
 # 
 # alldata<- clean.set(alldata)
 # 
 # #rm(training, validate, test)
 # 
 # gc()
 # 
 # alldata <- sparse(alldata)
# save(alldata, file = "alldata.RData")
load("alldata.RData")





training.set <- alldata[1:training.l,]
validate.set <- alldata[(training.l + 1):(training.l + validate.l),]
test.set <- alldata[(training.l + validate.l + 1):(nrow(alldata)),]
rm(alldata)
gc()


params <- list(
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

# 
# xgb.CV <- xgb.cv(data = X_train,
#                  params = params,
#                  early_stopping_rounds = 10,
#                  nfold = 3,
#                  nrounds = 10,
#                  feval = xg_eval_mae,
#                  maximize = FALSE
# )

gc()







# 
# 
# 
# 
# xgb1 <- xgboost(data = X_train, params = params, max_depth = 3, nrounds = 500)
# pred1 <- predict(xgb1, X_train)
# training.set$xgb1 <- pred1
# 
# gc()
# 
# pred1.v <- predict(xgb1, xgb.DMatrix(data.matrix(validate.set)))
# validate.set$xgb1 <- pred1.v
# 
# gc()
# 
# pred1.te <- predict(xgb1, xgb.DMatrix(data.matrix(test.set)))
# test.set$xgb1 <- pred1.te
# 
# 
# 
# gc()
# 
#  rf.tr <- randomForest(y=training.loss[1:1000], x=training.set[1:1000,], importance = TRUE, ntree = 200)
# # load("rf.tr.RData")
#  imp.rf <- varImp(rf.tr)
#  imp.rf$var <- rownames(imp.rf)
#  imp.rf <- c(arrange(imp.rf, desc(Overall))$var)
#  rf.v <- randomForest(y=validate.loss[1:1000], x=validate.set[1:1000,], importance = TRUE, ntree = 200)
# # load("rf.v.RData")
#  imp.v <- varImp(rf.v)
#  imp.v$var <- rownames(imp.v)
#  imp.v <- c(arrange(imp.v, desc(Overall))$var)
#  save(rf.tr, file = "rf.tr.RData")
#  save(rf.v, file = "rf.v.RData")
# 
# 
#  rf.vars <- unique(c(imp.rf[1:10], imp.v[1:10]))
# 
#  vars <- c()
#  for (i in 1:length(rf.vars)){
# 
#        subs <- names(training.set)[grep(gsub("_.*", "_", rf.vars[i]), names(training.set))]
#        rf.vars <- unique(c(rf.vars, subs))
#  }
#  gc()
#  training.rf.2 <- training.set[,rf.vars]
#  validate.rf.2 <- validate.set[,rf.vars]
#  test.rf.2 <- test.set[,rf.vars]
# 
#  save(training.rf.2, file = "training.rf.2.RData")
#  save(validate.rf.2, file = "validate.rf.2.RData")
#  save(test.rf.2, file = "test.rf.2.RData")

load("training.rf.2.RData")
load("validate.rf.2.RData")
load("test.rf.2.RData")


#
#
# create a some guesses at the loss of the test set with randomForest and our new rf sets.
#
#


# guess.ln <- randomForest(y=as.factor(training.ln), x=training.rf.2, importance = FALSE, prox = FALSE, ntree = 50)
# save(guess.ln, file = "guess.ln.RData")
load("guess.ln.RData")

# guess.th <- randomForest(y=as.factor(training.t), x=training.rf.2, importance = FALSE, prox = FALSE, ntree = 50)
# save(guess.th, file = "guess.th.RData")
load("guess.th.RData")

# pred.guess.ln <- predict(guess.ln, training.rf.2)
# save(pred.guess.ln, file = "pred.guess.ln.RData")
load("pred.guess.ln.RData")

# pred.guess.th <- predict(guess.th, training.rf.2)
# save(pred.guess.th, file = "pred.guess.th.RData")
load("pred.guess.th.RData")

gc()


# num.class.ln <- max(lns)-1
# 
# xgb.ln.params <- list(
#       booster = "gbtree",
#       objective = "multi:softmax",
#       num_class = num.class.ln,
#       eta = 0.05,
#       subsample = .7,
#       colsample_bytree = 0.7,
#       min_child_weight = 1,
#       num_parallel_tree = 1
# )
# nrounds.ln <- 50
# nfold.ln <- 4
# max.depth.ln <- 3
# 
# xtrain.ln <- xgb.DMatrix(data.matrix(training.set), label = data.matrix(training.ln))
# 
# xgb.ln <- xgboost(data=xtrain.ln, xgb.ln.params = params,
#                   nrounds = nrounds.ln,
#                   nfold = nfold.ln,
#                   max_depth = max.depth.ln)
# save(xgb.ln, file = "xgb.ln.RData")
load("xgb.ln.RData")

# pred.xgb.ln <- predict(xgb.ln, xtrain.ln)
# save(pred.xgb.ln, file = "pred.xgb.ln.RData")
load("pred.xgb.ln.RData")
# rm(xtrain.ln)

gc()

# num.class.th <- max(thous)-1
# 
# xgb.th.params <- list(
#       booster = "gbtree",
#       objective = "multi:softmax",
#       num_class = num.class.th,
#       eta = 0.05,
#       subsample = .7,
#       colsample_bytree = 0.7,
#       min_child_weight = 1,
#       num_parallel_tree = 1
# )
# nrounds.th <- 50
# nfold.th <- 4
# max.depth.th <- 3
# 
# xtrain.th <- xgb.DMatrix(data.matrix(training.set), label = data.matrix(training.t))
# 
# xgb.th <- xgboost(data=xtrain.th, xgb.th.params = params,
#                   nrounds = nrounds.th,
#                   nfold = nfold.th,
#                   max_depth = max.depth.th)
# save(xgb.th, file = "xgb.th.RData")
load("xgb.th.RData")


# pred.xgb.th <- predict(xgb.th, xtrain.th)
# save(pred.xgb.th, file = "pred.xgb.th.RData")
load("pred.xgb.th.RData")








########################################################################
                                ###stop###
########################################################################



alldata$ln <- factor(round(losses, 0))
alldata$round <- factor(thous)



rf.2 <- randomForest(y=training.loss, x=training.rf.2, importance = FALSE, prox = FALSE, ntree = 10)


pred.rf.2.t <- predict(rf.2, training.rf.2)
training.set$rf2 <- pred.rf.2.t

pred.rf.2.v <- predict(rf.2, validate.rf.2)
validate.set$rf2 <- pred.rf.2.v

pred.rf.2.te <- predict(rf.2, test.rf.2)
test.set$rf2 <- pred.rf.2.te


labels <- data.matrix(training.loss)
X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)


xgb1 <- xgboost(data = X_train, params = params, max_depth = 1, nrounds = 5)
pred1 <- predict(xgb1, xgb.DMatrix(data.matrix(training.set), label = training.loss))


