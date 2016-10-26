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



#training$thou <- round(training$loss, 3)




inTrain <- createDataPartition(training$loss, p=.7, list = FALSE)
training <- training[inTrain,]
validate <- training[-inTrain,]


#create new vectors with the loss amounts and ids so they can be removed from the training set
training.id <- training$id
validate.id <- validate$id
test.id <- test$id

training.loss <- log(training$loss)
validate.loss <- log(validate$loss)

training.l <- nrow(training)
validate.l <- nrow(validate)
test.l <- nrow(test)

training <- select(training, -id, -loss)
validate <- select(validate, -id, -loss)
test <- select(test, -id)

alldata <- rbind(training, validate, test)



nzv <- nearZeroVar(alldata)
alldata <- alldata[,-nzv]

clean.set <- function(input) {
      
      for (i in 1:length(grep("cat", names(input)))) {
            new.col <- input[,i]
            choices <- unique(new.col)
            for (k in 1:length(choices)) {
                  new.col[new.col == choices[k]] <- k
            }
            input[,i] <- as.factor(new.col)
      }
      input <- data.frame(input)
}

gc()

alldata<- clean.set(alldata)

rm(training, validate, test)

gc()

alldata <- sparse(alldata)

training.set <- alldata[1:training.l,]
validate.set <- alldata[(training.l + 1):(training.l + validate.l),]
test.set <- alldata[(training.l + validate.l + 1):(nrow(alldata)),]

gc()

labels <- data.matrix(training.loss)
X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)

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
#                  nfold = 4,
#                  nrounds = 750,
#                  feval = xg_eval_mae,
#                  maximize = FALSE
# )


# 
# training.rf <- training.set
# rf <- randomForest(y=training.loss[1:1000], x=training.rf[1:1000,], importance = TRUE, ntree = 1)
# imp <- varImp(rf)
# imp$var <- rownames(imp)
# imp <- c(arrange(imp, desc(Overall))$var[1:100])
# training.rf <- training.set[,imp]

gc()

rf.tr <- randomForest(y=training.loss[1:1000], x=training.set[1:1000,], importance = TRUE, ntree = 200)
imp.rf <- varImp(rf.tr)
imp.rf$var <- rownames(imp.tr)
imp.rf <- c(arrange(imp.rf, desc(Overall))$var)
rf.v <- randomForest(y=validate.loss[1:1000], x=validate.set[1:1000,], importance = TRUE, ntree = 200)
imp.v <- varImp(rf.v)
imp.v$var <- rownames(imp.v)
imp.v <- c(arrange(imp.v, desc(Overall))$var)

rf.vars <- unique(c(imp.tr[1:10], imp.v[1:10]))

vars <- c()
for (i in 1:length(rf.vars)){
      
      subs <- names(training.set)[grep(gsub("_.*", "_", rf.vars[i]), names(training.set))]
      rf.vars <- unique(c(rf.vars, subs))
}


training.rf.2 <- training.set[,rf.vars]
validate.rf.2 <- validate.set[,rf.vars]
test.rf.2 <- test.set[,rf.vars]

rf.2 <- randomForest(y=training.loss, x=training.rf.2, importance = FALSE, prox = FALSE, ntree = 10)


pred.rf.2.t <- predict(rf.2, training.rf)
pred.rf.2.v <- predict(rf.2, validate.rf.2)








