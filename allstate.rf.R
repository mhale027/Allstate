library(data.table)
library(randomForest)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)


setwd("~/Projects/Kaggle/Allstate")
set.seed(7890)

train <-fread("train.csv", header = TRUE)
test <- fread("test.csv", header = TRUE)
sample <- data.frame(fread("sample_submission.csv", header = TRUE))

#apply(train, 2, function(x) sum(is.na(x)))

#separate data for training into validation and training sets
#inTrain <- createDataPartition(train$loss, p=.7, list = FALSE)
#training <- train[inTrain,]
#validate <- train[-inTrain,]


#create new vectors with the loss amounts and ids so they can be removed from the training set
training.id <- train$id
#validate.id <- validate$id
test.id <- test$id

train <- select(train, -id)
test <- select(test, -id)

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
      
      output <- input
}

train <- clean.set(train)
test <- clean.set(test)

#train <- mutate(train, loss.cat = round(loss/2, -3)/1000)

train$cat <- NA
train$cat[train$loss < 2000] <- 0
train$cat[train$loss >= 2000 & train$loss < 4000] <- 1
train$cat[train$loss >= 4000] <- 2

train <- select(train, -loss)

rfst <- randomForest(factor(cat)~., train, ntree = 5, importance = TRUE)

pred.rf <- predict(rfst, test)


