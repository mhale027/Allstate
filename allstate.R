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

#apply(train, 2, function(x) sum(is.na(x)))

#separate data for training into validation and training sets
#inTrain <- createDataPartition(train$loss, p=.7, list = FALSE)
#training <- train[inTrain,]
#validate <- train[-inTrain,]


#create new vectors with the loss amounts and ids so they can be removed from the training set
training.id <- training$id
#validate.id <- validate$id
test.id <- test$id


training.loss <- log(training$loss)
#validate.loss <- validate$loss

training.data <- select(training, -id, -loss)
#validate.data <- select(validate, -id, -loss)
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
#validate.set <- clean.set(validate.data)
test.set <- clean.set(test.data)

labels <- as.matrix(training.loss)
data <- xgb.DMatrix(as.matrix(training.set), label = labels)

params <- list(
      #booster="gblinear", 
      objective="reg:linear", 
      eta=0.05, 
      max_depth=6, 
      subsample=.7, 
      colsample_bytree=0.7, 
      min_child_weight=1,
      base_score = 7,
      num_parallel_tree = 1
)

xg_eval_mae <- function (yhat, data) {
      labels = getinfo(data, "label")
      err= as.numeric(mean(abs(labels - yhat)))
      return (list(metric = "error", value = err))
}


xgb.CV <- xgb.cv(data = data, 
                 params = params, 
                 early_stopping_rounds = 15, 
                 nfold = 4, 
                 nrounds = 750,
                 feval = xg_eval_mae,
                 maximize = FALSE
                 )

boost <- xgboost(data = data, params = params, nrounds = xgb.CV$best_iteration)

pred.train <- predict(boost, as.matrix(training.set))
#pred.val <- predict(boost, as.matrix(validate.set))
pred.test <- predict(boost, as.matrix(test.set))


#rf <- randomForest(loss~., train, ntree = 5)


sample$loss = exp(pred.test)

write.csv(sample,'submission.10.20.2.csv',row.names = FALSE)

#id     loss
#1  4 1685.679
#2  6 1834.864
#3  9 7124.580
#4 12 5947.825
#5 15 1617.754
#6 17 2762.761





