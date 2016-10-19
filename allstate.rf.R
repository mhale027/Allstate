library(data.table)
library(randomForest)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)


setwd("~/Projects/Kaggle/Allstate")
set.seed(7890)

train <-data.frame(fread("train.csv", header = TRUE))
test <- data.frame(fread("test.csv", header = TRUE))
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

train <- clean.set(train)
test <- clean.set(test)

train <- mutate(train, loss.thou = round(loss/2, -3)/1000)

train$cat <- NA
train$cat[train$loss < 2000] <- 0
train$cat[train$loss >= 2000 & train$loss < 4000] <- 1
train$cat[train$loss >= 4000 & train$loss < 10000] <- 2
train$cat[train$loss >= 10000] <- 3
train$cat <- as.factor(train$cat)

train.loss.thou <- train$loss.thou
train.loss <- train$loss
train.cat <- train$cat
train <- select(train, -loss, -cat, -loss.thou)

labels <- data.matrix(train.cat)
dtrain <- xgb.DMatrix(data.matrix(train), label = labels)

xgb.params <- list(
      booster="gbtree", 
      objective="multi:softmax", 
      eta=0.1, 
      max_depth=6, 
      subsample=.7, 
      colsample_bytree=0.8, 
      min_child_weight=1,
      num_class = 4
)


#rfst.cat <- randomForest(y=as.factor(train.cat) , x = train, ntree = 5, importance = TRUE)
#xgb.cat <- xgboost(data = dtrain, params = xgb.params, nrounds = 50)
#pred.rf.cat <- predict(rfst, train)
#cm.rf.cat <- confusionMatrix(pred.rf, train.cat)
#pred.xgb.cat <- predict(xgb, data.matrix(train)l type = "response")
#cm.xgb.cat <- confusionMatrix(pred.xgb, train.cat)


#rfst.loss.thou <- randomForest(y=as.factor(train.loss.thou) , x = train, ntree = 5, importance = TRUE)
#xgb.loss.thou <- xgboost(data = dtrain, params = xgb.params, nrounds = 50)
#pred.rf.loss.thou <- predict(rfst, train)
#cm.rf.loss.thou <- confusionMatrix(pred.rf, train.loss.thou)
#pred.xgb.loss.thou <- predict(xgb, data.matrix(train)l type = "response")
#cm.xgb.loss.thou <- confusionMatrix(pred.xgb, train.loss.thou)






#rfst.loss <- randomForest(y=as.factor(train.loss) , x = train, ntree = 5, importance = TRUE)
#xgb.loss <- xgboost(data = dtrain, params = xgb.params, nrounds = 50)
#pred.rf.loss <- predict(rfst, train)
#cm.rf.loss <- confusionMatrix(pred.rf, train.loss)
#pred.xgb.loss <- predict(xgb, data.matrix(train)l type = "response")
#cm.xgb.loss <- confusionMatrix(pred.xgb, train.loss)