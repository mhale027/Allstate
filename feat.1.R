library(data.table)
library(randomForest)
library(ParallelForest)
library(ranger)
library(dplyr)
library(xgboost)
library(ggplot2)
library(caret)
library(Metrics)
library(nnet)
library(h2o)

setwd("~/Projects/kaggle/Allstate")
load("features.RData")
set.seed(7890)

training <- data.frame(fread("train.csv", header = TRUE))
test <- data.frame(fread("test.csv", header = TRUE))
sample <- data.frame(fread("sample_submission.csv", header = TRUE))

training.length <- nrow(training)
test.length <- nrow(test)

tr <- 1:training.length
te <- (1+training.length):(training.length + test.length)


training.id <- training$id
test.id <- test$id

training.loss <- training$loss


training <- select(training, -id, -loss)
test <- select(test, -id)
alldata <- rbind(training, test)
l <- c(training.loss, rep(NA, nrow(test)))
alldata <- cbind(l, alldata)
names(alldata)[1] <- "loss"


mean.loss <- mean(training.loss)


alldata <- data.table(alldata)
alldata[, c("cat12_cat80", "cat79_cat80", "cat57_cat80", "cat101_cat79", "cat57_cat79", "cat101_cat80"):=list(
      paste(cat12, cat80, sep="_"),
      paste(cat79, cat80, sep="_"),
      paste(cat57, cat80, sep="_"),
      paste(cat101, cat79, sep="_"),
      paste(cat57, cat79, sep="_"),
      paste(cat101, cat80, sep="_")
)]

alldata[, c("cat101_cat79_cat81", "cat57_cat79_cat80", "cat103_cat12_cat80"):=list(
      paste(cat101, cat79, cat81, sep="_"),
      paste(cat57, cat79, cat80, sep="_"),
      paste(cat103, cat12, cat80, sep="_")
)]


cols = names(alldata)

for (f in cols) {
      if (class(alldata[[f]])=="character") {
            #cat("VARIABLE : ",f,"\n")
            levels <- unique(alldata[[f]])
            alldata[[f]] <- as.integer(factor(alldata[[f]], levels=levels))
      }
}

alldata <- data.frame(alldata)
feat.1 <- rep(0, nrow(alldata))
# feat.2 <- rep(0, nrow(alldata))
# feat.3 <- rep(0, nrow(alldata))
# feat.4 <- rep(0, nrow(alldata))
# feat.5 <- rep(0, nrow(alldata))
# feat.6 <- rep(0, nrow(alldata))


feat.1[tr] <- ifelse(training.loss > mean.loss, 1, 0)

# 
# 
# z.score <- (training.loss - mean.loss)/sd(training.loss)
# for (i in tr) {
#       if (z.score[i] < -.5) {
#             feat.2[i] <- 1
#       } else if (z.score[i] < .02) {
#             feat.2[i] <- 2
#       } else {
#             feat.2[i] <- 3
#       }
# }



alldata$loss <- as.factor(feat.1)
names(alldata)[1] <- "feat.1"
levels(alldata$feat.1) <- make.names(levels(alldata$feat.1), unique = TRUE)


x.vars <- names(alldata)[-1]
response <- names(alldata)[1]

cat.vars <- names(alldata)[grep("cat", names(alldata))]
cont.vars <- names(alldata)[grep("cont", names(alldata))]









ytrain <- as.numeric(factor(alldata$feat.1))-1
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))






xgb.grid <- expand.grid(
      nrounds = 10,
      eta = 0.1,
      max_depth = c(8, 10, 12, 14),
      gamma = c(0, 3, 5),
      colsample_bytree = c(0.7, 0.85, 1.0),
      min_child_weight = c(0, 1, 2)
)

xgb.train.control <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      summaryFunction = twoClassSummary,
      allowParallel = TRUE
)


xgb.prep <- train(x=alldata[1:1000,-1],
                  y=alldata[1:1000,1],
                  trControl = xgb.train.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree",
                  metric = "ROC"
                  
)



xgb.1.max_depth <- xgb.prep$bestTune$max_depth
xgb.1.gamma <- xgb.prep$bestTune$gamma
xgb.1.colsample_bytree <- xgb.prep$bestTune$colsample_bytree
xgb.1.min_child_weight <- xgb.prep$bestTune$min_child_weight

xgb.1.params <- list(
      objective = "binary:logistic",
      # num_class = 2,
      eta = 0.2,
      max_depth = xgb.1.max_depth, 
      gamma = xgb.prep$bestTune$gamma,
      colsample_bytree = xgb.prep$bestTune$colsample_bytree,
      min_child_weight = xgb.prep$bestTune$min_child_weight    
)

res.1 <- xgb.cv(data = xtrain,
                nrounds = 1000,
                nfold = 4,
                params = xgb.1.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)
nrounds.1 <- res.1$best_iteration                
xgb.1 <- xgboost(xtrain, params = xgb.1.params, nfold = 4, nrounds = nrounds.1, early_stopping_rounds = 10)

pred.train.xgb.1 <- as.numeric(predict(xgb.1, xtrain) > .5)
pred.test.xgb.1 <- as.numeric(predict(xgb.1, xtest) > .5)

head(pred.train.xgb.1) 
head(pred.test.xgb.1)


guess.1.1<- data.frame(guess.1.1 = c(pred.train.xgb.1, pred.test.xgb.1))
alldata <- cbind(alldata, guess.1.1)

xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])

xgb.2 <- xgboost(xtrain, params = xgb.1.params, nfold = 4, nrounds = 500, early_stopping_rounds = 10)

pred.train.xgb.2 <- as.numeric(predict(xgb.2, xtrain) > .5)
pred.test.xgb.2 <- as.numeric(predict(xgb.2, xtest) > .5)

sum(abs(feat.1[tr] - pred.train.xgb.2))    #26254
table(feat.1[tr], pred.train.xgb.2)

guess.1.2 <- data.frame(guess.1.2 = c(pred.train.xgb.2, pred.test.xgb.2))
alldata <- cbind(alldata, guess.1.2)

xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])

xgb.3 <- xgboost(xtrain, params = xgb.1.params, nfold = 4, nrounds = 500, early_stopping_rounds = 10)

pred.train.xgb.3 <- as.numeric(predict(xgb.3, xtrain) > .5)
pred.test.xgb.3 <- as.numeric(predict(xgb.3, xtest) > .5)

sum(abs(feat.1[tr] - pred.train.xgb.3))    #26254
table(feat.1[tr], pred.train.xgb.3)

guess.1.3 <- data.frame(guess.1.3 = c(pred.train.xgb.3, pred.test.xgb.3))
alldata <- cbind(alldata, guess.1.3)

rf <- train(feat.1~ guess.1.1 + guess.1.2 + guess.1.3, alldata, ntree = 100)











pred.rf <- predict(rf, select(alldata, guess.1.1, guess.1.2, guess.1.3)[tr,])

table(feat.1, as.numeric(pred.rf)-1)

