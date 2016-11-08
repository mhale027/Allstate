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


sparse.frame <- function(train) {
      train <- data.table(train)
      c.vars <- names(train)
      for (i in 1:length(c.vars)) {
            gc()
            factors <- unique(train[[c.vars[i]]])
            new.cols <- paste(c.vars[i], factors, sep = "_")
            for (k in 1:length(factors)) {
                  new.c <- new.cols[k]
                  train[[new.c]] <- ifelse(train[[c.vars[i]]] == factors[k], 1, 0)
            }
            train[,c.vars[i]:=NULL]
      }
      train <- data.frame(train)
}

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
feat.1 <- rep(1, nrow(alldata))
feat.2 <- rep(1, nrow(alldata))
feat.3 <- rep(1, nrow(alldata))
feat.4 <- rep(1, nrow(alldata))
feat.5 <- rep(1, nrow(alldata))
feat.6 <- rep(1, nrow(alldata))
z.score <- c((training.loss - mean.loss)/sd(training.loss), rep(0, test.length))



feat.1[tr] <- ifelse(training.loss > mean.loss, 1, 0)


for (i in tr) {
      if (z.score[i] < -.5) {
            feat.2[i] <- 1
      } else if (z.score[i] < -.02) {
            feat.2[i] <- 2
      } else {
            feat.2[i] <- 3
      }
}


for (i in tr) {
      if (z.score[i] < -.68) {
            feat.3[i] <- 1
      } else if (z.score[i] < -.46) {
            feat.3[i] <- 2
      }else if (z.score[i] < -.13) {
            feat.3[i] <- 3
      }else if (z.score[i] < .49) {
            feat.3[i] <- 4
      } else {
            feat.3[i] <- 5
      }
}


for (i in tr) {
      if (z.score[i] < -.74) {
            feat.4[i] <- 1
      } else if (z.score[i] < -.59) {
            feat.4[i] <- 2
      }else if (z.score[i] < -.42) {
            feat.4[i] <- 3
      }else if (z.score[i] < -.19) {
            feat.4[i] <- 4
      }else if (z.score[i] < .16) {
            feat.4[i] <- 5
      }else if (z.score[i] < .81) {
            feat.4[i] <- 6
      } else {
            feat.4[i] <- 7
      }
}


for (i in tr) {
      if (z.score[i] < -.77) {
            feat.5[i] <- 1
      } else if (z.score[i] < -.66) {
            feat.5[i] <- 2
      }else if (z.score[i] < -.54) {
            feat.5[i] <- 3
      }else if (z.score[i] < -.40) {
            feat.5[i] <- 4
      }else if (z.score[i] < -.22) {
            feat.5[i] <- 5
      }else if (z.score[i] < .22) {
            feat.5[i] <- 6
      }else if (z.score[i] < .39) {
            feat.5[i] <- 7
      }else if (z.score[i] < 1.06) {
            feat.5[i] <- 8
      } else {
            feat.5[i] <- 9
      }
}


for (i in tr) {
      if (z.score[i] < -.79) {
            feat.6[i] <- 1
      } else if (z.score[i] < -.70) {
            feat.6[i] <- 2
      }else if (z.score[i] < -.61) {
            feat.6[i] <- 3
      }else if (z.score[i] < -.50) {
            feat.6[i] <- 4
      }else if (z.score[i] < -.39) {
            feat.6[i] <- 5
      }else if (z.score[i] < -.24) {
            feat.6[i] <- 6
      }else if (z.score[i] < -.05) {
            feat.6[i] <- 7
      }else if (z.score[i] < .20) {
            feat.6[i] <- 8
      }else if (z.score[i] < .58) {
            feat.6[i] <- 9
      }else if (z.score[i] < 1.25) {
            feat.6[i] <- 10
      } else {
            feat.6[i] <- 11
      }
}



feat.1 <- data.frame(feat.1 = as.factor(feat.1))
feat.2 <- data.frame(feat.2 = as.factor(feat.2))
feat.3<- data.frame(feat.3 = as.factor(feat.3))
feat.4 <- data.frame(feat.4 = as.factor(feat.4))
feat.5 <- data.frame(feat.5 = as.factor(feat.5))
feat.6 <- data.frame(feat.6 = as.factor(feat.6))
feat.z <- data.frame(feat.z = z.score)

feats <- cbind(feat.1, feat.2, feat.3, feat.4, feat.5, feat.6)
# feats <- sparse.frame(feats)
feats <- cbind(feats, feat.z)

# alldata <- cbind(alldata, feats)

levels(feat.1$feat.1) <- make.names(levels(feat.1$feat.1), unique = TRUE)
levels(feat.2$feat.2) <- make.names(levels(feat.2$feat.2), unique = TRUE)
levels(feat.3$feat.3) <- make.names(levels(feat.3$feat.3), unique = TRUE)
levels(feat.4$feat.4) <- make.names(levels(feat.4$feat.4), unique = TRUE)
levels(feat.5$feat.5) <- make.names(levels(feat.5$feat.5), unique = TRUE)
levels(feat.6$feat.6) <- make.names(levels(feat.6$feat.6), unique = TRUE)

save(alldata, file = "template.RData")
# save(nzv.t, file = "nzv.t.RData")





#################################################################
##################################################################
####################       feature guess 1        ###############
################################################################
###############################################################

# Fitting 
# nrounds = 10, 
# max_depth = 10, 
# eta = 0.1, 
# gamma = 5, 
# colsample_bytree = 0.7, 
# min_child_weight = 2

ytrain <- as.numeric(factor(feat.1[tr,]))-1
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))

xgb.grid.1 <- expand.grid(
      nrounds = 10,
      eta = 0.1,
      max_depth = c(8, 10, 12, 14),
      gamma = c(0, 3, 5),
      colsample_bytree = c(0.7, 0.85, 1.0),
      min_child_weight = c(0, 1, 2)
)

xgb.train.control.1 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      summaryFunction = twoClassSummary,
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.1 <- train(x = alldata[1:1000,-1],
                  y=feat.1[1:1000,],
                  trControl = xgb.train.control.1,
                  tuneGrid = xgb.grid.1,
                  method = "xgbTree",
                  metric = "ROC"
                  
)



xgb.1.max_depth <- xgb.prep.1$bestTune$max_depth
xgb.1.gamma <- xgb.prep.1$bestTune$gamma
xgb.1.colsample_bytree <- xgb.prep.1$bestTune$colsample_bytree
xgb.1.min_child_weight <- xgb.prep.1$bestTune$min_child_weight

xgb.1.params <- list(
      objective = "binary:logistic",
      eta = 0.2,
      max_depth = xgb.1.max_depth, 
      gamma = xgb.1.gamma,
      colsample_bytree = xgb.1.colsample_bytree,
      min_child_weight = xgb.1.min_child_weight,
      metric = "ROC"
)

res.1 <- xgb.cv(data = xtrain,
                nrounds = 500,
                nfold = 4,
                params = xgb.1.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.1 <- res.1$best_iteration     

xgb.1 <- xgboost(xtrain, params = xgb.1.params,
                 nfold = 4, 
                 nrounds = nrounds.1, 
                 early_stopping_rounds = 10)

pred.train.xgb.1 <- as.numeric(predict(xgb.1, xtrain) > .5)
pred.test.xgb.1 <- as.numeric(predict(xgb.1, xtest) > .5)

# table(pred.train.xgb.1, feat.1[tr,])   pred.train.xgb.1     X0     X1
                                                        # 0 116489  16705
                                                        # 1   7390  47734

xgb.grid.2 <- expand.grid(
      nrounds = 10,
      eta = 0.05,
      max_depth = c(9, 10, 11, 12),
      gamma = c(2, 3, 4),
      colsample_bytree = c(0.7, 0.75, .8),
      min_child_weight = 1
)

xgb.train.control.2 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      summaryFunction = twoClassSummary,
      allowParallel = TRUE
)


xgb.prep.2 <- train(x = alldata[1:1000,-1],
                  y=feat.1[1:1000,],
                  trControl = xgb.train.control.2,
                  tuneGrid = xgb.grid.2,
                  method = "xgbTree",
                  metric = "ROC"
                  
)



xgb.2.max_depth <- xgb.prep.2$bestTune$max_depth
xgb.2.gamma <- xgb.prep.2$bestTune$gamma
xgb.2.colsample_bytree <- xgb.prep.2$bestTune$colsample_bytree
xgb.2.min_child_weight <- xgb.prep.2$bestTune$min_child_weight

xgb.2.params <- list(
      objective = "binary:logistic",
      # num_class = 2,
      eta = 0.05,
      max_depth = xgb.2.max_depth, 
      gamma = xgb.2.gamma,
      colsample_bytree = xgb.2.colsample_bytree,
      min_child_weight = xgb.2.min_child_weight,
      metric = "ROC"
)


res.2 <- xgb.cv(data = xtrain,
                nrounds = 1000,
                nfold = 4,
                params = xgb.2.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.2 <- res.2$best_iteration  

xgb.2 <- xgboost(xtrain, nrounds = nrounds.2,
                 params = xgb.2.params,
                 early_stopping_rounds = 5,
                 nfold = 4
)

pred.train.xgb.2 <- as.numeric(predict(xgb.2, xtrain) > .5)
pred.test.xgb.2 <- as.numeric(predict(xgb.2, xtest) > .5)

#no ptrain
# table(pred.train.xgb.2, feat.1)   X0     X1
                               # 0 117857  15791
                               # 1   6022  48648

f.1 <- data.frame(f.1 = c(pred.train.xgb.2, pred.test.xgb.2))
save(f.1, file = "feature.1.RData")
alldata <- cbind(alldata, f.1)


#################################################################
##################################################################
####################       feature guess 2        ###############
################################################################
###############################################################

# Fitting 
# nrounds = 10, 
# max_depth = 10, 
# eta = 0.1, 
# gamma = 5, 
# colsample_bytree = 0.7, 
# min_child_weight = 2

ytrain <- as.numeric(factor(feat.2[tr,]))-1
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))

xgb.grid.1 <- expand.grid(
      nrounds = 10,
      eta = 0.1,
      max_depth = c(8, 10, 12, 14),
      gamma = c(0, 3, 5),
      colsample_bytree = c(0.7, 0.85, 1.0),
      min_child_weight = c(0, 1, 2)
)

xgb.train.control.1 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.1 <- train(x = alldata[1:1000,-1],
                    y=feat.2[1:1000,],
                    trControl = xgb.train.control.1,
                    tuneGrid = xgb.grid.1,
                    method = "xgbTree",
                    metric = "Accuracy"
                    
)



xgb.1.max_depth <- xgb.prep.1$bestTune$max_depth
xgb.1.gamma <- xgb.prep.1$bestTune$gamma
xgb.1.colsample_bytree <- xgb.prep.1$bestTune$colsample_bytree
xgb.1.min_child_weight <- xgb.prep.1$bestTune$min_child_weight

xgb.1.params <- list(
      objective = "multi:softmax",
      num_class = 3,
      eta = 0.2,
      max_depth = xgb.1.max_depth, 
      gamma = xgb.1.gamma,
      colsample_bytree = xgb.1.colsample_bytree,
      min_child_weight = xgb.1.min_child_weight,
      metric = "Accuracy"
)

res.1 <- xgb.cv(data = xtrain,
                nrounds = 500,
                nfold = 4,
                params = xgb.1.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.1 <- res.1$best_iteration     

xgb.1 <- xgboost(xtrain, params = xgb.1.params,
                 nfold = 4, 
                 nrounds = nrounds.1, 
                 early_stopping_rounds = 10)

pred.train.xgb.1 <- predict(xgb.1, xtrain)
pred.test.xgb.1 <- predict(xgb.1, xtest)

# table(pred.train.xgb.1, feat.1[tr,])   pred.train.xgb.1     X0     X1
# 0 116489  16705
# 1   7390  47734

xgb.grid.2 <- expand.grid(
      nrounds = 10,
      eta = 0.05,
      max_depth = c(9, 10, 11, 12),
      gamma = c(2, 3, 4),
      colsample_bytree = c(0.7, 0.75, .8),
      min_child_weight = 1
)

xgb.train.control.2 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      summaryFunction = twoClassSummary,
      allowParallel = TRUE
)


xgb.prep.2 <- train(x = alldata[1:1000,-1],
                    y=feat.1[1:1000,],
                    trControl = xgb.train.control.2,
                    tuneGrid = xgb.grid.2,
                    method = "xgbTree",
                    metric = "Accuracy"
                    
)



xgb.2.max_depth <- xgb.prep.2$bestTune$max_depth
xgb.2.gamma <- xgb.prep.2$bestTune$gamma
xgb.2.colsample_bytree <- xgb.prep.2$bestTune$colsample_bytree
xgb.2.min_child_weight <- xgb.prep.2$bestTune$min_child_weight

xgb.2.params <- list(
      objective = "binary:logistic",
      # num_class = 2,
      eta = 0.05,
      max_depth = xgb.2.max_depth, 
      gamma = xgb.2.gamma,
      colsample_bytree = xgb.2.colsample_bytree,
      min_child_weight = xgb.2.min_child_weight,
      metric = "mlogloss"
)


res.2 <- xgb.cv(data = xtrain,
                nrounds = 1000,
                nfold = 4,
                params = xgb.2.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.2 <- res.2$best_iteration  

xgb.2 <- xgboost(xtrain, nrounds = nrounds.2,
                 params = xgb.2.params,
                 early_stopping_rounds = 5,
                 nfold = 4
)

pred.train.xgb.2 <- predict(xgb.2, xtrain)
pred.test.xgb.2 <- predict(xgb.2, xtest)

# table(as.numeric(factor(feat.2$feat.2[tr])), pred)    ### no f.1 in alldata ###
# pred
# 1     2     3
# 1 56393  8530  4326
# 2 18382 23540 11236
# 3  6135  7810 51966
f.2 <- data.frame(f.2 = c(pred.train.xgb.2, pred.test.xgb.2))
save(f.2, file = "feature.2.RData")

##############################  END  ###############################







ptrain <- predict(xgb.1, xtrain, outputmargin = TRUE)
ptest <- predict(xgb.1, xtrain, outputmargin = TRUE)

setinfo(xtrain, "base_margin", ptrain)
setinfo(xtest, "base_margin", ptest)

# 1   7390  47734

xgb.grid.2 <- expand.grid(
      nrounds = 10,
      eta = 0.05,
      max_depth = c(9, 10, 11, 12),
      gamma = c(2, 3, 4),
      colsample_bytree = c(0.7, 0.75, .8),
      min_child_weight = 1
)

xgb.train.control.2 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      summaryFunction = twoClassSummary,
      allowParallel = TRUE
)


xgb.prep.2 <- train(x = alldata[1:1000,-1],
                    y=feat.1[1:1000,],
                    trControl = xgb.train.control.2,
                    tuneGrid = xgb.grid.2,
                    method = "xgbTree",
                    metric = "ROC"
                    
)



xgb.2.max_depth <- xgb.prep.2$bestTune$max_depth
xgb.2.gamma <- xgb.prep.2$bestTune$gamma
xgb.2.colsample_bytree <- xgb.prep.2$bestTune$colsample_bytree
xgb.2.min_child_weight <- xgb.prep.2$bestTune$min_child_weight

xgb.2.params <- list(
      objective = "binary:logistic",
      # num_class = 2,
      eta = 0.05,
      max_depth = xgb.2.max_depth, 
      gamma = xgb.2.gamma,
      colsample_bytree = xgb.2.colsample_bytree,
      min_child_weight = xgb.2.min_child_weight,
      metric = "mlogloss"
)


res.2 <- xgb.cv(data = xtrain,
                nrounds = 1000,
                nfold = 4,
                params = xgb.2.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.2 <- res.2$best_iteration  

xgb.2 <- xgboost(xtrain, nrounds = nrounds.2,
                 params = xgb.2.params,
                 early_stopping_rounds = 5,
                 nfold = 4
)

pred.train.xgb.2.1 <- as.numeric(predict(xgb.2, xtrain) > .5)
pred.test.xgb.2.1 <- as.numeric(predict(xgb.2, xtest) > .5)

table(pred.train.xgb.2.1, feat.1[tr,])
