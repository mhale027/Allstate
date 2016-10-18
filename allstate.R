library(data.table)
library(randomForest)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)







setwd("~/Projects/Kaggle/Allstate")

train <-fread("train.csv", header = TRUE)
test <- fread("test.csv", header = TRUE)
sample <- data.frame(fread("sample_submission.csv", header = TRUE))

#apply(train, 2, function(x) sum(is.na(x)))

inTrain <- createDataPartition(train$loss, p=.7, list = FALSE)
training <- train[inTrain,]
validate <- train[-inTrain,]



training.id <- training$id
validate.id <- validate$id
test.id <- test$id


training.loss <- training$loss
validate.loss <- validate$loss

training.data <- select(training, -id, -loss)
validate.data <- select(validate, -id, -loss)
test.data <- select(test, -id)

clean.set <- function(input) {

      for (i in 1:116) {
            
            new.col <- rep(NA, nrow(input))
            choices <- unique(input[,i])
            
            for (k in 1:length(choices)) {
                  
                  new.col[input[,i] == choices[k]] <- k
                  
            }
      
            input[,i] <- as.numeric(new.col)
      
      }
      
      output <- input
}

training.set <- clean.set(training.data)
validate.set <- clean.set(validate.data)
test.set <- clean.set(test.data)

labels <- as.matrix(training.loss)
data <- xgb.DMatrix(as.matrix(training.set), label = labels)

params <- list(
      booster="gblinear", 
      objective="reg:linear", 
      eta=0.05, 
      max_depth=6, 
      subsample=1, 
      colsample_bytree=0.5, 
      min_child_weight=1
)



boost <- xgboost(data = data, params = params, nrounds = 100)

pred.train <- predict(boost, as.matrix(training.set))
pred.val <- predict(boost, as.matrix(validate.set))
pred.test <- predict(boost, as.matrix(test.set))


#rf <- randomForest(loss~., train.clean, ntree = 5)


sample$loss = exp(pred.test)

write.csv(sample,'xgb.rip1.test.csv',row.names = FALSE)
