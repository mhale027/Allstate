library(data.table)
library(randomForest)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(Metrics)


setwd("~/Projects/Kaggle/Allstate")
set.seed(7890)

training <-data.frame(fread("train.csv", header = TRUE))
test <- data.frame(fread("test.csv", header = TRUE))
sample <- data.frame(fread("sample_submission.csv", header = TRUE))


# training <- training[1:10000,]

#create new vectors with the loss amounts and ids so they can be removed from the training set
training.id <- training$id
test.id <- test$id


training.loss <- log(training$loss)



training.data <- select(training, -id, -loss)
test.data <- select(test, -id)



clean.set <- function(input) {
      
      for (i in 1:116) {
            new.col <- input[,i]
            choices <- unique(new.col)
            for (k in 1:length(choices)) {
                  new.col[new.col == choices[k]] <- k
            }
            input[,i] <- as.numeric(new.col)
      }
      input <- input
}

training.set <- clean.set(training.data)
test.set <- clean.set(test.data)




labels <- data.matrix(training.loss)
X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)

params <- list(
      booster = "gblinear",
      objective = "reg:linear", 
      eta = 0.05, 
      subsample = .7, 
      colsample_bytree = 0.7, 
      min_child_weight = 1,
      base_score = 7,
      num_parallel_tree = 1
)

xg_eval_mae <- function (yhat, data) {
      labels = getinfo(data, "label")
      err= as.numeric(mean(abs(labels - yhat)))
      return (list(metric = "error", value = err))
}


xgb.CV <- xgb.cv(data = X_train,
                 params = params,
                 early_stopping_rounds = 10,
                 nfold = 4,
                 nrounds = 750,
                 feval = xg_eval_mae,
                 maximize = FALSE
                 )

nrounds <- xgb.CV$best_iteration

#boost <- xgboost(data = X_train, params = params, nrounds = xgb.CV$best_iteration)



# 
# pred.train <- exp(predict(boost, as.matrix(training.set)))
# pred.val <- exp(predict(boost, as.matrix(validate.set)))
# pred.test <- exp(predict(boost, as.matrix(test.set)))


# rf <- randomForest(loss~., train, ntree = 5)


# sample$loss = pred.test
# 
# mod <- function(data, labels, model) {
#       data <- data.matrix(data)
#       labels <- data.matrix(labels)
#       pred <- predict(model, xgb.DMatrix(data, labels))
#       
#       
# }

rf1 <- randomForest(y=training.loss, x=training.set, ntree = 50, importance = TRUE)
pred.rf <- predict(rf1, training.set)
training.set$rf1 <- pred.rf
      
      
X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb1 <- xgboost(data = X_train, params = params, max_depth = 1, nrounds = nrounds)
pred1 <- predict(xgb1, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred1 <- pred1

X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb2 <- xgboost(data = X_train, params = params, max_depth = 2, nrounds = nrounds)
pred2 <- pred2 <- predict(xgb2, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred2 <- pred2

X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb3 <- xgboost(data = X_train, params = params, max_depth = 3, nrounds = nrounds)
pred3 <- pred3 <- predict(xgb3, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred3 <- pred3

X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb4 <- xgboost(data = X_train, params = params, max_depth = 4, nrounds = nrounds)
pred4 <- pred4 <- predict(xgb4, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred4 <- pred4

X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb5 <- xgboost(data = X_train, params = params, max_depth = 5, nrounds = nrounds)
pred5 <- pred5 <- predict(xgb5, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred5 <- pred5

X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb6 <- xgboost(data = X_train, params = params, max_depth = 6, nrounds = nrounds)
pred6 <- pred6 <- predict(xgb6, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred6 <- pred6

# write.csv(sample,'submission.10.20.2.csv',row.names = FALSE)

#id     loss
#1  4 1685.679
#2  6 1834.864
#3  9 7124.580
#4 12 5947.825
#5 15 1617.754
#6 17 2762.761

# 
# id     loss
# 1  4 1654.149
# 2  6 1836.539
# 3  9 7772.634
# 4 12 5345.297
# 5 15 1420.574
# 6 17 2714.712
# 
# validated set
# 
# 
# # 


# gblinear
# id     loss
# 1  4 1602.187
# 2  6 1664.112
# 3  9 6229.759
# 4 12 4296.716
# 5 15 1371.976
# 6 17 2180.190