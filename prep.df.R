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



#create new vectors with the loss amounts and ids so they can be removed from the training set
training.id <- training$id
test.id <- test$id
ids <- c(training.id, test.id)


training.thou <- round(training$loss/1000)
test.thou <- round(test$loss/1000)
thous <- c(training.thou, test.thou)




training.loss <- log(training$loss)
test.loss <- log(test$loss)
losses <- c(training.loss, test.loss)

training.ln <- round(training.loss)
test.ln <- round(test.loss)
lns <- c(training.ln, test.ln)



training.l <- nrow(training)
test.l <- nrow(test)
lengths <- c(training.l, test.l)

training <- select(training, -id, -loss)
test <- select(test, -id, -loss)

gc()

# alldata <- rbind(training, test)
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
# #rm(training, test)
# 
# gc()
# 
# alldata <- sparse(alldata)
# save(alldata, file = "alldata.0.RData")
load("alldata.0.RData")





training.set <- alldata[1:training.l,]
test.set <- alldata[(training.l + 1):(nrow(alldata)),]
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








gc()

# rf.tr <- randomForest(y=training.loss[1:1000], x=training.set[1:1000,], importance = TRUE, ntree = 200)
load("rf.tr.0.RData")
imp.rf <- varImp(rf.tr)
imp.rf$var <- rownames(imp.rf)
imp.rf <- c(arrange(imp.rf, desc(Overall))$var)
# rf.te <- randomForest(y=test.loss[1:1000], x=test.set[1:1000,], importance = TRUE, ntree = 200)
# # load("rf.te.0.RData")
# imp.te <- varImp(rf.te)
# imp.te$var <- rownames(imp.te)
# imp.te <- c(arrange(imp.te, desc(Overall))$var)
# save(rf.tr, file = "rf.tr.0.RData")
# save(rf.v, file = "rf.te.0.RData")
# 
# 
# rf.vars <- unique(c(imp.rf[1:20]))
# vars <- c()
# for (i in 1:length(rf.vars)){
# subs <- names(training.set)[grep(gsub("_.*", "_", rf.vars[i]), names(training.set))]
# rf.vars <- unique(c(rf.vars, subs))
# }
# 
# gc()
# 
# 
# training.rf.2 <- training.set[,rf.vars]
# test.rf.2 <- test.set[,rf.vars]
# 
# save(training.rf.2, file = "training.rf.2.0.RData")
# save(test.rf.2, file = "test.rf.2.0.RData")

load("training.rf.2.0.RData")
load("test.rf.2.0.RData")


#
#
# create a some guesses at the loss of the test set with randomForest and our new rf sets.
#
#


# guess.ln <- randomForest(y=as.factor(training.ln), x=training.rf.2, importance = FALSE, prox = FALSE, ntree = 50)
# save(guess.ln, file = "guess.ln.0.RData")
load("guess.ln.0.RData")

# guess.th <- randomForest(y=as.factor(training.thou), x=training.rf.2, importance = FALSE, prox = FALSE, ntree = 50)
# save(guess.th, file = "guess.th.0.RData")
load("guess.th.0.RData")

# pred.guess.ln <- predict(guess.ln, training.rf.2)
# save(pred.guess.ln, file = "pred.guess.ln.0.RData")
load("pred.guess.ln.0.RData")

# pred.guess.th <- predict(guess.th, training.rf.2)
# save(pred.guess.th, file = "pred.guess.th.0.RData")
load("pred.guess.th.0.RData")

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
# 
# nrounds.ln <- 50
# nfold.ln <- 4
# max.depth.ln <- 3
# 
# xtrain.ln <- xgb.DMatrix(data.matrix(training.set), label = data.matrix(training.ln))
# 
# xgb.ln <- xgboost(data=xtrain.ln, xgb.ln.params = params,
#              nrounds = nrounds.ln,
#              nfold = nfold.ln,
#              max_depth = max.depth.ln)
# save(xgb.ln, file = "xgb.ln.0.RData")
load("xgb.ln.0.RData")

gc()


# pred.xgb.ln <- predict(xgb.ln, xtrain.ln)
# save(pred.xgb.ln, file = "pred.xgb.ln.0.RData")
load("pred.xgb.ln.0.RData")
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
# xtrain.th <- xgb.DMatrix(data.matrix(training.set), label = data.matrix(training.thou))
# 
# xgb.th <- xgboost(data=xtrain.th, xgb.th.params = params,
#                   nrounds = nrounds.th,
#                   nfold = nfold.th,
#                   max_depth = max.depth.th)
# save(xgb.th, file = "xgb.th.0.RData")
load("xgb.th.0.RData")

# 
# pred.xgb.th <- predict(xgb.th, xtrain.th)
# save(pred.xgb.th, file = "pred.xgb.th.0.RData")
load("pred.xgb.th.0.RData")

ytrain <- xgb.DMatrix(data.matrix(test.set))



pred.xgb.th <- as.numeric(predict(xgb.th, ytrain))
pred.xgb.ln <- as.numeric(predict(xgb.ln, ytrain))
pred.guess.th <- as.numeric(predict(guess.th, test.rf.2))
pred.guess.ln <- as.numeric(predict(guess.th, test.rf.2))


test.ln.guess <- round((pred.xgb.ln+ pred.guess.ln)/2)
test.th.guess <- round((pred.xgb.th+ pred.guess.th)/2)


test.set$ln <- test.ln.guess
test.set$thou <- test.th.guess

training.set$ln <- training.ln
training.set$thou <- training.thou


alldata <- rbind(training.set, test.set)
save(alldata, file = "alldata.final.RData")

features <- list(c("lengths", "ids", "losses", "lns", "thous"), lengths, ids, losses, lns, thous)
save(features, file = "features.RData")
