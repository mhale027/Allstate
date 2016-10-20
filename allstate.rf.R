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
inTrain <- createDataPartition(train$loss, p=.7, list = FALSE)
training <- train[inTrain,]
validate <- train[-inTrain,]


#create new vectors with the loss amounts and ids so they can be removed from the training set
training.id <- train$id
validate.id <- validate$id
test.id <- test$id

training <- select(training, -id)
validate <- select(validate, -id)
testing <- select(test, -id)

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

training <- clean.set(training)
validate <- clean.set(validate)
testing <- clean.set(testing)

training <- mutate(training, loss.thou = round(loss/2, -3)/1000)

training$cat <- NA
training$cat[training$loss < 2000] <- 0
training$cat[training$loss >= 2000 & training$loss < 4000] <- 1
training$cat[training$loss >= 4000 & training$loss < 10000] <- 2
training$cat[training$loss >= 10000] <- 3
training$cat <- as.factor(training$cat)

training.loss.thou <- training$loss.thou
training.loss <- training$loss
training.cat <- training$cat
training <- select(training, -loss, -cat, -loss.thou)

validate <- select(validate, -loss)

labels <- data.matrix(training.cat)
dtrain <- xgb.DMatrix(data.matrix(training), label = labels)

xgb.params <- list(
      booster="gbtree", 
      objective="multi:softmax", 
      eta=0.05, 
      max_depth=6, 
      subsample=.7, 
      colsample_bytree=0.7, 
      min_child_weight=1,
      num_class = 4
)

rf.params <- list(
      ntree = 500,
      verbose = TRUE,
      importance = TRUE,
      nodesize = 1,
      prox = TRUE
)

#rfst.cat <- randomForest(y=as.factor(training.cat) , x = training, ntree = 5, importance = TRUE)
#xgb.cat <- xgboost(data = dtrain, params = xgb.params, nrounds = 50)
#pred.rf.cat <- predict(rfst.cat, training)
#cm.rf.cat <- confusionMatrix(pred.rf.cat, training.cat)
#pred.xgb.cat <- predict(xgb.cat, data.matrix(training), type = "response")
#cm.xgb.cat <- confusionMatrix(pred.xgb.cat, training.cat)


#rfst.loss.thou <- randomForest(y=as.factor(training.loss.thou) , x = training, ntree = 5, importance = TRUE)
#xgb.loss.thou <- xgboost(data = dtrain, params = xgb.params, nrounds = 50)
#pred.rf.loss.thou <- predict(rfst.loss.thou, training)
#cm.rf.loss.thou <- confusionMatrix(pred.rf.loss.thou, training.loss.thou)
#pred.xgb.loss.thou <- predict(xgb.loss.thou, data.matrix(training), type = "response")
#cm.xgb.loss.thou <- confusionMatrix(pred.xgb.loss.thou, training.loss.thou)


#rfst.loss <- randomForest(y=as.factor(training.loss) , x = training, ntree = 5, importance = TRUE)
#xgb.loss <- xgboost(data = dtrain, params = xgb.params, nrounds = 50)
#pred.rf.loss <- predict(rfst.loss, training)
#cm.rf.loss <- confusionMatrix(pred.rf.loss, training.loss)
#pred.xgb.loss <- predict(xgb.loss, data.matrix(training), type = "response")
#cm.xgb.loss <- confusionMatrix(pred.xgb.loss, training.loss)