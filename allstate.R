library(data.table)
library(randomForest)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)


setwd("~/Projects/Kaggle/Allstate")
set.seed(7890)

training <-fread("train.csv", header = TRUE)
test <- fread("test.csv", header = TRUE)
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


training.loss <- training$loss
#validate.loss <- validate$loss

training.data <- select(training, -id, -loss)
#validate.data <- select(validate, -id, -loss)
test.data <- select(test, -id)

clean.set <- function(input) {
      n <- nrow(input)
      for (i in 1:116) {
            new.col <- rep(NA, n)
            choices <- unique(input[,i])
            for (k in 1:length(choices)) {
                  new.col[input[,i] == choices[k]] <- k
            }
            input[,i] <- as.numeric(new.col)
      }
      
      output <- as.matrix(input)
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

#xgb.CV <- xgb.cv(data = data, params = params, early_stopping_rounds = 20, nfolds = 3, nrounds = 3000)

boost <- xgboost(data = data, params = params, nrounds = 1000)

pred.train <- predict(boost, as.matrix(training.set))
#pred.val <- predict(boost, as.matrix(validate.set))
pred.test <- predict(boost, as.matrix(test.set))


#rf <- randomForest(loss~., train, ntree = 5)


sample$loss = pred.test

write.csv(sample,'submission.2.csv',row.names = FALSE)
