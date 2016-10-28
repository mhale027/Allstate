library(data.table)
library(randomForest)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(Metrics)


setwd("~/Projects/Kaggle/Allstate")
set.seed(7890)

load("alldata.final.RData")
load("features.RData")

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

lns <- features[[5]]
training.ln <- lns[tr]
test.ln <- lns[te]

thous <- features[[6]]
training.thou <- thous[tr]
test.thou <- thous[te]




training.set <- alldata[1:training.length,]
test.set <- alldata[(training.length + 1):(nrow(alldata)),]
rm(alldata)

rm(features, lengths, ids, losses, lns, thous)

gc()




rf.tr <- randomForest(y=training.loss[1:1000], x=training.set[1:1000,], importance = TRUE, ntree = 200)
imp.rf <- varImp(rf.tr)
imp.rf$var <- rownames(imp.rf)
imp.rf <- c(arrange(imp.rf, desc(Overall))$var)
save(rf.tr, file = "rf.tr.00.RData")
# load("rf.tr.RData")


rf.vars <- unique(c(imp.rf[1:20]))

vars <- c()

for (i in 1:length(rf.vars)){
 subs <- names(training.set)[grep(gsub("_.*", "_", rf.vars[i]), names(training.set))]
 rf.vars <- unique(c(rf.vars, subs))
}

gc()

training.rf.1 <- training.set[,rf.vars]
test.rf.1 <- test.set[,rf.vars]

save(training.rf.1, file = "training.rf.1.00.RData")
save(test.rf.1, file = "test.rf.1.00.RData")

# load("training.rf.1.00.RData")
# load("test.rf.1.00.RData")


gc()



controls.1 <- trainControl(method = "repeadedcv",
                           number = 5,
                           repeat = 3)

mtry.1 = floor(sqrt(ncol(training.rf.1)))
ntrees.1 = 500


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











