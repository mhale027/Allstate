library(data.table)
library(randomForest)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)




set.seed(7890)


setwd("~/Projects/Kaggle/Allstate")

train <- data.frame(fread("train.csv", header = TRUE))
test <- data.frame(fread("test.csv", header = TRUE))
sample <- data.frame(fread("sample_submission.csv", header = TRUE))

#apply(train, 2, function(x) sum(is.na(x)))

#inTrain <- createDataPartition(train$loss, p=.7, list = FALSE)
training <- train
#validate <- train[-inTrain,]



training.id <- training$id
#validate.id <- validate$id
test.id <- test$id


training.loss <- log(training$loss)
#validate.loss <- validate$loss


training <- mutate(training, cat = round(loss/1000,0))


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
      
            input[,i] <- as.factor(new.col)
      
      }
      
      input <- input
}

training.set <- clean.set(training.data)
#validate.set <- clean.set(validate.data)
test.set <- clean.set(test.data)

labels <- data.matrix(training.loss)
data <- xgb.DMatrix(data.matrix(training.set), label = labels)

params <- list(
      booster="gblinear", 
      objective="reg:linear", 
      eta=0.05, 
      max_depth=6, 
      subsample=1, 
      colsample_bytree=0.75, 
      min_child_weight=1
)



boost <- xgboost(data = data, params = params, nrounds = 100)

pred.train <- predict(boost, data.matrix(training.set))
#pred.val <- predict(boost, data.matrix(validate.set))
pred.test <- predict(boost, data.matrix(test.set))


#rf <- randomForest(loss~., train.clean, ntree = 5)


sample$loss = exp(pred.test)

write.csv(sample,'xgb.test3.csv',row.names = FALSE)
