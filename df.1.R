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

# training <- filter(training, loss < 11500)

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
feat_1 <- rep(1, nrow(alldata))
feat_2 <- rep(1, nrow(alldata))
feat_3 <- rep(1, nrow(alldata))
feat_4 <- rep(1, nrow(alldata))
feat_5 <- rep(1, nrow(alldata))
feat_6 <- rep(1, nrow(alldata))
z.score <- c((training.loss - mean.loss)/sd(training.loss), rep(0, test.length))
z <- sort(z.score[tr])


feat_1[tr] <- ifelse(training.loss > mean.loss, 1, 0)


for (i in tr) {
      if (z.score[i] < -.5) {
            feat_2[i] <- 1
      } else if (z.score[i] < .5) {
            feat_2[i] <- 2
      } else {
            feat_2[i] <- 3
      }
}


for (i in tr) {
      if (z.score[i] < -.6) {
            feat_3[i] <- 1
      } else if (z.score[i] < 0) {
            feat_3[i] <- 2
      }else if (z.score[i] < 1) {
            feat_3[i] <- 3
      }else {
            feat_3[i] <- 4
      } 
}


for (i in tr) {
      if (z.score[i] < -.9) {
            feat_4[i] <- 1
      } else if (z.score[i] < -.3) {
            feat_4[i] <- 2
      }else if (z.score[i] < 0) {
            feat_4[i] <- 3
      }else if (z.score[i] < .02) {
            feat_4[i] <- 4
      }else if (z.score[i] < .7) {
            feat_4[i] <- 5
      }else if (z.score[i] < 2) {
            feat_4[i] <- 6
      } else {
            feat_4[i] <- 7
      }
}


for (i in tr) {
      if (z.score[i] < -1) {
            feat_5[i] <- 1
      } else if (z.score[i] < -.8) {
            feat_5[i] <- 2
      }else if (z.score[i] < -.5) {
            feat_5[i] <- 3
      }else if (z.score[i] < -.30) {
            feat_5[i] <- 4
      }else if (z.score[i] < -.22) {
            feat_5[i] <- 5
      }else if (z.score[i] < .1) {
            feat_5[i] <- 6
      }else if (z.score[i] < .49) {
            feat_5[i] <- 7
      }else if (z.score[i] < 1.16) {
            feat_5[i] <- 8
      } else {
            feat_5[i] <- 9
      }
}


for (i in tr) {
      if (z.score[i] < -1) {
            feat_6[i] <- 1
      } else if (z.score[i] < -.9) {
            feat_6[i] <- 2
      }else if (z.score[i] < -.7) {
            feat_6[i] <- 3
      }else if (z.score[i] < -.60) {
            feat_6[i] <- 4
      }else if (z.score[i] < -.45) {
            feat_6[i] <- 5
      }else if (z.score[i] < -.2) {
            feat_6[i] <- 6
      }else if (z.score[i] < .05) {
            feat_6[i] <- 7
      }else if (z.score[i] < .30) {
            feat_6[i] <- 8
      }else if (z.score[i] < .58) {
            feat_6[i] <- 9
      }else if (z.score[i] < 1.5) {
            feat_6[i] <- 10
      } else {
            feat_6[i] <- 11
      }
}

feats <- data.frame(feat_1=feat_1, feat_2=feat_2, feat_3=feat_3, feat_4=feat_4,
                    feat_5=feat_5, feat_6=feat_6, feat.z = z.score)

feat_1 <- data.frame(feat_1 = as.factor(feat_1))
feat_2 <- data.frame(feat_2 = as.factor(feat_2))
feat_3 <- data.frame(feat_3 = as.factor(feat_3))
feat_4 <- data.frame(feat_4 = as.factor(feat_4))
feat_5 <- data.frame(feat_5 = as.factor(feat_5))
feat_6 <- data.frame(feat_6 = as.factor(feat_6))
feat_z <- data.frame(feat_z = z.score)

levels(feat_1$feat_1) <- make.names(levels(feat_1$feat_1), unique = TRUE)
levels(feat_2$feat_2) <- make.names(levels(feat_2$feat_2), unique = TRUE)
levels(feat_3$feat_3) <- make.names(levels(feat_3$feat_3), unique = TRUE)
levels(feat_4$feat_4) <- make.names(levels(feat_4$feat_4), unique = TRUE)
levels(feat_5$feat_5) <- make.names(levels(feat_5$feat_5), unique = TRUE)
levels(feat_6$feat_6) <- make.names(levels(feat_6$feat_6), unique = TRUE)



save(alldata, file = "template.RData")
# save(nzv.t, file = "nzv.t.RData")
gc()

#################################################################
##################################################################
####################       feature guess 1        ###############
################################################################
###############################################################

# Fitting 
# nrounds = 10, 
# max_depth = 10, 
# eta = .2, 
# gamma = 5, 
# colsample_bytree = 0.7, 
# min_child_weight = 2

ytrain <- as.numeric(factor(feat_1[tr,]))-1
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))

xgb.grid.f1.1 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(8, 10, 12, 14),
      gamma = c(0, 3, 5),
      colsample_bytree = c(0.7, 0.85, 1.0),
      min_child_weight = c(0, 1, 2)
)

xgb.train.control.f1.1 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f1.1 <- train(x = alldata[1:1000,-1],
                       y=feat_1[1:1000,],
                       trControl = xgb.train.control.f1.1,
                       tuneGrid = xgb.grid.f1.1,
                       method = "xgbTree",
                       metric = "Accuracy"
)



xgb.f1.1.max_depth <- xgb.prep.f1.1$bestTune$max_depth
xgb.f1.1.gamma <- xgb.prep.f1.1$bestTune$gamma
xgb.f1.1.colsample_bytree <- xgb.prep.f1.1$bestTune$colsample_bytree
xgb.f1.1.min_child_weight <- xgb.prep.f1.1$bestTune$min_child_weight

xgb.f1.1.params <- list(
      objective = "multi:softmax",
      num_class = 2,
      eta = 0.2,
      max_depth = xgb.f1.1.max_depth, 
      gamma = xgb.f1.1.gamma,
      colsample_bytree = xgb.f1.1.colsample_bytree,
      min_child_weight = xgb.f1.1.min_child_weight,
      metric = "Accuracy"
)

res.f1.1 <- xgb.cv(data = xtrain,
                   nrounds = 500,
                   nfold = 4,
                   params = xgb.f1.1.params,
                   early_stopping_rounds = 10,
                   print_every_n = 10
)

nrounds.f1.1 <- res.f1.1$best_iteration     

xgb.f1.1 <- xgboost(xtrain, params = xgb.f1.1.params,
                    nfold = 4, 
                    nrounds = nrounds.f1.1, 
                    early_stopping_rounds = 10)

pred.train.xgb.f1.1 <- predict(xgb.f1.1, xtrain)
pred.test.xgb.f1.1 <- predict(xgb.f1.1, xtest)

# table(pred.train.xgb.1, feat_1[tr,])   pred.train.xgb.1     X0     X1
# 0 116489  16705
# 1   7390  47734

xgb.grid.f1.2 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(9, 10, 11, 12),
      gamma = c(2, 3, 4),
      colsample_bytree = c(0.7, 0.75, .8),
      min_child_weight = 1
)

xgb.train.control.f1.2 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f1.2 <- train(x = alldata[1:1000,-1],
                       y=feat_1[1:1000,],
                       trControl = xgb.train.control.f1.2,
                       tuneGrid = xgb.grid.f1.2,
                       method = "xgbTree",
                       metric = "Accuracy"
                       
)



xgb.f1.2.max_depth <- xgb.prep.f1.2$bestTune$max_depth
xgb.f1.2.gamma <- xgb.prep.f1.2$bestTune$gamma
xgb.f1.2.colsample_bytree <- xgb.prep.f1.2$bestTune$colsample_bytree
xgb.f1.2.min_child_weight <- xgb.prep.f1.2$bestTune$min_child_weight

xgb.f1.2.params <- list(
      objective = "multi:softmax",
      num_class = 2,
      eta = .2,
      max_depth = xgb.f1.2.max_depth, 
      gamma = xgb.f1.2.gamma,
      colsample_bytree = xgb.f1.2.colsample_bytree,
      min_child_weight = xgb.f1.2.min_child_weight,
      metric = "Accuracy"
)


res.f1.2 <- xgb.cv(data = xtrain,
                   nrounds = 1000,
                   nfold = 4,
                   params = xgb.f1.2.params,
                   early_stopping_rounds = 10,
                   print_every_n = 10
)

nrounds.f1.2 <- res.f1.2$best_iteration  

xgb.f1.2 <- xgboost(xtrain, nrounds = nrounds.f1.2,
                    params = xgb.f1.2.params,
                    early_stopping_rounds = 5,
                    nfold = 4
)

pred.train.xgb.f1.2 <- predict(xgb.f1.2, xtrain)
pred.test.xgb.f1.2 <- predict(xgb.f1.2, xtest)

#no ptrain
# table(pred.train.xgb.2, feat_1)   X0     X1
# 0 117857  15791
# 1   6022  48648

# filtered z.score under 3


f.1 <- data.frame(f.1 = c(pred.train.xgb.f1.2, pred.test.xgb.f1.2))

save(f.1_1, file = "feature.1.RData")
save(xgb.f1_1, file = "xgb.f1.2.RData")
alldata <- cbind(alldata, f.1)




#################################################################
##################################################################
####################       feature guess 2        ###############
################################################################
###############################################################
gc()
# Fitting 
# nrounds = 10, 
# max_depth = 10, 
# eta = .2, 
# gamma = 5, 
# colsample_bytree = 0.7, 
# min_child_weight = 2

ytrain <- as.numeric(factor(feat_2[tr,]))-1
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))

xgb.grid.f2.1 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(8, 10, 12, 14),
      gamma = c(0, 3, 5),
      colsample_bytree = c(0.7, 0.85, 1.0),
      min_child_weight = c(0, 1, 2)
)

xgb.train.control.f2.1 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f2.1 <- train(x = alldata[1:1000,-1],
                  y=feat_2[1:1000,],
                  trControl = xgb.train.control.f2.1,
                  tuneGrid = xgb.grid.f2.1,
                  method = "xgbTree",
                  metric = "Accuracy"
                  
)



xgb.f2.1.max_depth <- xgb.prep.f2.1$bestTune$max_depth
xgb.f2.1.gamma <- xgb.prep.f2.1$bestTune$gamma
xgb.f2.1.colsample_bytree <- xgb.prep.f2.1$bestTune$colsample_bytree
xgb.f2.1.min_child_weight <- xgb.prep.f2.1$bestTune$min_child_weight

xgb.f2.1.params <- list(
      objective = "multi:softmax",
      num_class = 3,
      eta = 0.2,
      max_depth = xgb.f2.1.max_depth, 
      gamma = xgb.f2.1.gamma,
      colsample_bytree = xgb.f2.1.colsample_bytree,
      min_child_weight = xgb.f2.1.min_child_weight,
      metric = "Accuracy"
)

res.f2.1 <- xgb.cv(data = xtrain,
                nrounds = 500,
                nfold = 4,
                params = xgb.f2.1.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.f2.1 <- res.f2.1$best_iteration     

xgb.f2.1 <- xgboost(xtrain, params = xgb.f2.1.params,
                 nfold = 4, 
                 nrounds = nrounds.f2.1, 
                 early_stopping_rounds = 10)

pred.train.xgb.f2.1 <- predict(xgb.f2.1, xtrain)
pred.test.xgb.f2.1 <- predict(xgb.f2.1, xtest)

# table(pred.train.xgb.1, feat_1[tr,])   pred.train.xgb.1     X0     X1
                                                        # 0 116489  16705
                                                        # 1   7390  47734

xgb.grid.f2.2 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(9, 10, 11, 12),
      gamma = c(2, 3, 4),
      colsample_bytree = c(0.7, 0.75, .8),
      min_child_weight = 1
)

xgb.train.control.f2.2 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f2.2 <- train(x = alldata[1:1000,-1],
                  y=feat_2[1:1000,],
                  trControl = xgb.train.control.f2.2,
                  tuneGrid = xgb.grid.f2.2,
                  method = "xgbTree",
                  metric = "Accuracy"
                  
)



xgb.f2.2.max_depth <- xgb.prep.f2.2$bestTune$max_depth
xgb.f2.2.gamma <- xgb.prep.f2.2$bestTune$gamma
xgb.f2.2.colsample_bytree <- xgb.prep.f2.2$bestTune$colsample_bytree
xgb.f2.2.min_child_weight <- xgb.prep.f2.2$bestTune$min_child_weight

xgb.f2.2.params <- list(
      objective = "multi:softmax",
      num_class = 3,
      eta = .2,
      max_depth = xgb.f2.2.max_depth, 
      gamma = xgb.f2.2.gamma,
      colsample_bytree = xgb.f2.2.colsample_bytree,
      min_child_weight = xgb.f2.2.min_child_weight,
      metric = "Accuracy"
)


res.f2.2 <- xgb.cv(data = xtrain,
                nrounds = 1000,
                nfold = 4,
                params = xgb.f2.2.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.f2.2 <- res.f2.2$best_iteration  

xgb.f2.2 <- xgboost(xtrain, nrounds = nrounds.f2.2,
                 params = xgb.f2.2.params,
                 early_stopping_rounds = 5,
                 nfold = 4
)

pred.train.xgb.f2.2 <- predict(xgb.f2.2, xtrain) 
pred.test.xgb.f2.2 <- predict(xgb.f2.2, xtest) 

#no ptrain
# table(pred.train.xgb.2, feat_1)   X0     X1
                               # 0 117857  15791
                               # 1   6022  48648

# filtered z.score under 3


f.2 <- data.frame(f.2 = c(pred.train.xgb.f2.2, pred.test.xgb.f2.2))

save(f.2_1, file = "feature.2.RData")
save(xgb.f2.2_1, file = "xgb.f2.2.RData")
alldata <- cbind(alldata, f.2)

#################################################################
##################################################################
####################       feature guess 3        ###############
################################################################
###############################################################
gc()
# Fitting 
# nrounds = 10, 
# max_depth = 10, 
# eta = .2, 
# gamma = 5, 
# colsample_bytree = 0.7, 
# min_child_weight = 2

ytrain <- as.numeric(factor(feat_3[tr,]))-1
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))

xgb.grid.f3.1 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(8, 10, 12, 14),
      gamma = c(0, 3, 5),
      colsample_bytree = c(0.7, 0.85, 1.0),
      min_child_weight = c(0, 1, 2)
)

xgb.train.control.f3.1 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f3.1 <- train(x = alldata[1:1000,-1],
                    y=feat_3[1:1000,],
                    trControl = xgb.train.control.f3.1,
                    tuneGrid = xgb.grid.f3.1,
                    method = "xgbTree",
                    metric = "Accuracy"
                    
)



xgb.f3.1.max_depth <- xgb.prep.f3.1$bestTune$max_depth
xgb.f3.1.gamma <- xgb.prep.f3.1$bestTune$gamma
xgb.f3.1.colsample_bytree <- xgb.prep.f3.1$bestTune$colsample_bytree
xgb.f3.1.min_child_weight <- xgb.prep.f3.1$bestTune$min_child_weight

xgb.f3.1.params <- list(
      objective = "multi:softmax",
      num_class = 4,
      eta = 0.2,
      max_depth = xgb.f3.1.max_depth, 
      gamma = xgb.f3.1.gamma,
      colsample_bytree = xgb.f3.1.colsample_bytree,
      min_child_weight = xgb.f3.1.min_child_weight,
      metric = "Accuracy"
)

res.f3.1 <- xgb.cv(data = xtrain,
                nrounds = 500,
                nfold = 4,
                params = xgb.f3.1.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.f3.1 <- res.f3.1$best_iteration     

xgb.f3.1 <- xgboost(xtrain, params = xgb.f3.1.params,
                 nfold = 4, 
                 nrounds = nrounds.f3.1, 
                 early_stopping_rounds = 10)

pred.train.xgb.f3.1 <- predict(xgb.f3.1, xtrain)
pred.test.xgb.f3.1 <- predict(xgb.f3.1, xtest)

# table(pred.train.xgb.f3.1, feat_1[tr,])   pred.train.xgb.f3.1     X0     X1
# 0 116489  16705
# 1   7390  47734

xgb.grid.f3.2 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(9, 10, 11, 12),
      gamma = c(2, 3, 4),
      colsample_bytree = c(0.7, 0.75, .8),
      min_child_weight = 1
)

xgb.train.control.f3.2 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f3.2 <- train(x = alldata[1:1000,-1],
                    y=feat_3[1:1000,],
                    trControl = xgb.train.control.f3.2,
                    tuneGrid = xgb.grid.f3.2,
                    method = "xgbTree",
                    metric = "Accuracy"
                    
)



xgb.f3.2.max_depth <- xgb.prep.f3.2$bestTune$max_depth
xgb.f3.2.gamma <- xgb.prep.f3.2$bestTune$gamma
xgb.f3.2.colsample_bytree <- xgb.prep.f3.2$bestTune$colsample_bytree
xgb.f3.2.min_child_weight <- xgb.prep.f3.2$bestTune$min_child_weight

xgb.f3.2.params <- list(
      objective = "multi:softmax",
      num_class = 4,
      eta = .2,
      max_depth = xgb.f3.2.max_depth, 
      gamma = xgb.f3.2.gamma,
      colsample_bytree = xgb.f3.2.colsample_bytree,
      min_child_weight = xgb.f3.2.min_child_weight,
      metric = "Accuracy"
)


res.f3.2 <- xgb.cv(data = xtrain,
                nrounds = 1000,
                nfold = 4,
                params = xgb.f3.2.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.f3.2 <- res.f3.2$best_iteration  

xgb.f3.2 <- xgboost(xtrain, nrounds = nrounds.f3.2,
                 params = xgb.f3.2.params,
                 early_stopping_rounds = 5,
                 nfold = 4
)

pred.train.xgb.f3.2 <- predict(xgb.f3.2, xtrain)
pred.test.xgb.f3.2 <- predict(xgb.f3.2, xtest)

# table(as.numeric(factor(feat_2$feat_2[tr])), pred)    ### no f.f3.1 in alldata ###
# pred
# 1     2     3
# 1 56393  8530  4326
# 2 18382 23540 11236
# 3  6135  7810 51966

# pred.train.xgb.f3.2
# 0     1     2
# 1 57983  9837  1429
# 2 19018 29161  4979
# 3  7157 10409 48345


# filtered
# pred.train.xgb.f3.2             X1    X2    X3
#                        0  66697 18449  9543
#                        1   1733 11409  1930
#                        2   6229 11183 57358


# filtered and adjusted around z.score = 0
# pred.train.xgb.f3.2       pred.train.xgb.f3.2
#                             0      1      2
#                             1 110234      0   6140
#                             2    822      7    454
#                             3  15101      0  51773

f.3 <- data.frame(f.3 = c(pred.train.xgb.f3.2, pred.test.xgb.f3.2))
save(f.3_1, file = "feature.3.RData")
save(xgb.f3.2_1, file = "xgb.f3.2.RData")
alldata <- cbind(alldata, f.3)






#################################################################
##################################################################
####################       feature guess 4        ###############
################################################################
###############################################################
gc()
# Fitting 
# nrounds = 10, 
# max_depth = 10, 
# eta = .2, 
# gamma = 5, 
# colsample_bytree = 0.7, 
# min_child_weight = 2

ytrain <- as.numeric(factor(feat_4[tr,]))-1
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))

xgb.grid.f4.1 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(8, 10, 12, 14),
      gamma = c(0, 3, 5),
      colsample_bytree = c(0.7, 0.85, 1.0),
      min_child_weight = c(0, 1, 2)
)

xgb.train.control.f4.1 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f4.1 <- train(x = alldata[1:1000,-1],
                    y=feat_4[1:1000,],
                    trControl = xgb.train.control.f4.1,
                    tuneGrid = xgb.grid.f4.1,
                    method = "xgbTree",
                    metric = "Accuracy"
                    
)



xgb.f4.1.max_depth <- xgb.prep.f4.1$bestTune$max_depth
xgb.f4.1.gamma <- xgb.prep.f4.1$bestTune$gamma
xgb.f4.1.colsample_bytree <- xgb.prep.f4.1$bestTune$colsample_bytree
xgb.f4.1.min_child_weight <- xgb.prep.f4.1$bestTune$min_child_weight

xgb.f4.1.params <- list(
      objective = "multi:softmax",
      num_class = 7,
      eta = 0.2,
      max_depth = xgb.f4.1.max_depth, 
      gamma = xgb.f4.1.gamma,
      colsample_bytree = xgb.f4.1.colsample_bytree,
      min_child_weight = xgb.f4.1.min_child_weight,
      metric = "Accuracy"
)

res.f4.1 <- xgb.cv(data = xtrain,
                nrounds = 500,
                nfold = 4,
                params = xgb.f4.1.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.f4.1 <- res.f4.1$best_iteration     

xgb.f4.1 <- xgboost(xtrain, params = xgb.f4.1.params,
                 nfold = 4, 
                 nrounds = nrounds.f4.1, 
                 early_stopping_rounds = 10)

pred.train.xgb.f4.1 <- predict(xgb.f4.1, xtrain)
pred.test.xgb.f4.1 <- predict(xgb.f4.1, xtest)

# table(pred.train.xgb.f4.1, feat_1[tr,])   pred.train.xgb.f4.1     X0     X1
# 0 116489  16705
# 1   7390  47734

xgb.grid.f4.2 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(9, 10, 11, 12),
      gamma = c(2, 3, 4),
      colsample_bytree = c(0.7, 0.75, .8),
      min_child_weight = 1
)

xgb.train.control.f4.2 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f4.2 <- train(x = alldata[1:1000,-1],
                    y=feat_4[1:1000,],
                    trControl = xgb.train.control.f4.2,
                    tuneGrid = xgb.grid.f4.2,
                    method = "xgbTree",
                    metric = "Accuracy"
                    
)



xgb.f4.2.max_depth <- xgb.prep.f4.2$bestTune$max_depth
xgb.f4.2.gamma <- xgb.prep.f4.2$bestTune$gamma
xgb.f4.2.colsample_bytree <- xgb.prep.f4.2$bestTune$colsample_bytree
xgb.f4.2.min_child_weight <- xgb.prep.f4.2$bestTune$min_child_weight

xgb.f4.2.params <- list(
      objective = "multi:softmax",
      num_class = 7,
      eta = .2,
      max_depth = xgb.f4.2.max_depth, 
      gamma = xgb.f4.2.gamma,
      colsample_bytree = xgb.f4.2.colsample_bytree,
      min_child_weight = xgb.f4.2.min_child_weight,
      metric = "Accuracy"
)


res.f4.2 <- xgb.cv(data = xtrain,
                nrounds = 1000,
                nfold = 4,
                params = xgb.f4.2.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.f4.2 <- res.f4.2$best_iteration  

xgb.f4.2 <- xgboost(xtrain, nrounds = nrounds.f4.2,
                 params = xgb.f4.2.params,
                 early_stopping_rounds = 5,
                 nfold = 4
)

pred.train.xgb.f4.2 <- predict(xgb.f4.2, xtrain)
pred.test.xgb.f4.2 <- predict(xgb.f4.2, xtest)

# table(as.numeric(factor(feat_2$feat_2[tr])), pred)    ### no f.1 in alldata ###
# pred
# 1     2     3
# 1 56393  8530  4326
# 2 18382 23540 11236
# 3  6135  7810 51966

# pred.train.xgb.f4.2
# 0     1     2
# 1 57983  9837  1429
# 2 19018 29161  4979
# 3  7157 10409 48345


# filtered
# pred.train.xgb.f4.2             X1    X2    X3
#                        0  66697 18449  9543
#                        1   1733 11409  1930
#                        2   6229 11183 57358


# filtered and adjusted around z.score = 0
# pred.train.xgb.f4.2       pred.train.xgb.f4.2
#                             0      1      2
#                             1 110234      0   6140
#                             2    822      7    454
#                             3  15101      0  51773

f.4 <- data.frame(f.4 = c(pred.train.xgb.f4.2, pred.test.xgb.f4.2))
save(f.4_1, file = "feature.4.RData")
save(xgb.f4.2_1, file = "xgb.f4.2.RData")
alldata <- cbind(alldata, f.4)




#################################################################
##################################################################
####################       feature guess 5        ###############
################################################################
###############################################################
gc()
# Fitting 
# nrounds = 10, 
# max_depth = 10, 
# eta = .2, 
# gamma = 5, 
# colsample_bytree = 0.7, 
# min_child_weight = 2

ytrain <- as.numeric(factor(feat_5[tr,]))-1
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))

xgb.grid.f5.1 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(8, 10, 12, 14),
      gamma = c(0, 3, 5),
      colsample_bytree = c(0.7, 0.85, 1.0),
      min_child_weight = c(0, 1, 2)
)

xgb.train.control.f5.1 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f5.1 <- train(x = alldata[1:1000,-1],
                       y=feat_5[1:1000,],
                       trControl = xgb.train.control.f5.1,
                       tuneGrid = xgb.grid.f5.1,
                       method = "xgbTree",
                       metric = "Accuracy"
                       
)



xgb.f5.1.max_depth <- xgb.prep.f5.1$bestTune$max_depth
xgb.f5.1.gamma <- xgb.prep.f5.1$bestTune$gamma
xgb.f5.1.colsample_bytree <- xgb.prep.f5.1$bestTune$colsample_bytree
xgb.f5.1.min_child_weight <- xgb.prep.f5.1$bestTune$min_child_weight

xgb.f5.1.params <- list(
      objective = "multi:softmax",
      num_class = 9,
      eta = 0.2,
      max_depth = xgb.f5.1.max_depth, 
      gamma = xgb.f5.1.gamma,
      colsample_bytree = xgb.f5.1.colsample_bytree,
      min_child_weight = xgb.f5.1.min_child_weight,
      metric = "Accuracy"
)

res.f5.1 <- xgb.cv(data = xtrain,
                   nrounds = 500,
                   nfold = 4,
                   params = xgb.f5.1.params,
                   early_stopping_rounds = 10,
                   print_every_n = 10
)

nrounds.f5.1 <- res.f5.1$best_iteration     

xgb.f5.1 <- xgboost(xtrain, params = xgb.f5.1.params,
                    nfold = 4, 
                    nrounds = nrounds.f5.1, 
                    early_stopping_rounds = 10)

pred.train.xgb.f5.1 <- predict(xgb.f5.1, xtrain)
pred.test.xgb.f5.1 <- predict(xgb.f5.1, xtest)

# table(pred.train.xgb.f5.1, feat_5[tr,])   pred.train.xgb.f5.1     X0     X1
#                                                                 0 116489  16705
#                                                                 1   7390  47734

xgb.grid.f5.2 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(9, 10, 11, 12),
      gamma = c(2, 3, 4),
      colsample_bytree = c(0.7, 0.75, .8),
      min_child_weight = 1
)

xgb.train.control.f5.2 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f5.2 <- train(x = alldata[1:1000,-1],
                    y=feat_5[1:1000,],
                    trControl = xgb.train.control.f5.2,
                    tuneGrid = xgb.grid.f5.2,
                    method = "xgbTree",
                    metric = "Accuracy"
                    
)



xgb.f5.2.max_depth <- xgb.prep.f5.2$bestTune$max_depth
xgb.f5.2.gamma <- xgb.prep.f5.2$bestTune$gamma
xgb.f5.2.colsample_bytree <- xgb.prep.f5.2$bestTune$colsample_bytree
xgb.f5.2.min_child_weight <- xgb.prep.f5.2$bestTune$min_child_weight

xgb.f5.2.params <- list(
      objective = "multi:softmax",
      num_class = 9,
      eta = .2,
      max_depth = xgb.f5.2.max_depth, 
      gamma = xgb.f5.2.gamma,
      colsample_bytree = xgb.f5.2.colsample_bytree,
      min_child_weight = xgb.f5.2.min_child_weight,
      metric = "Accuracy"
)


res.f5.2 <- xgb.cv(data = xtrain,
                nrounds = 1000,
                nfold = 4,
                params = xgb.f5.2.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.f5.2 <- res.f5.2$best_iteration  

xgb.f5.2 <- xgboost(xtrain, nrounds = nrounds.f5.2,
                 params = xgb.f5.2.params,
                 early_stopping_rounds = 5,
                 nfold = 4
)

pred.train.xgb.f5.2 <- predict(xgb.f5.2, xtrain)
pred.test.xgb.f5.2 <- predict(xgb.f5.2, xtest)

# table(as.numeric(factor(feat_5$feat_5[tr])), pred)    ### no f.1 in alldata ###
# pred
# 1     2     3
# 1 56393  8530  4326
# 2 18382 23540 11236
# 3  6135  7810 51966

# pred.train.xgb.2
# 0     1     2
# 1 57983  9837  1429
# 2 19018 29161  4979
# 3  7157 10409 48345


# filtered
# pred.train.xgb.2             X1    X2    X3
#                        0  66697 18449  9543
#                        1   1733 11409  1930
#                        2   6229 11183 57358


# filtered and adjusted around z.score = 0
# pred.train.xgb.2       pred.train.xgb.2
#                             0      1      2
#                             1 110234      0   6140
#                             2    822      7    454
#                             3  15101      0  51773

f.5 <- data.frame(f.5 = c(pred.train.xgb.f5.2, pred.test.xgb.f5.2))
save(f.5_1, file = "feature.5.RData")
save(xgb.f5.2_1, file = "xgb.f5.2.RData")
alldata <- cbind(alldata, f.5)



#################################################################
##################################################################
####################       feature guess 6        ###############
################################################################
###############################################################
gc()
# Fitting 
# nrounds = 10, 
# max_depth = 10, 
# eta = .2, 
# gamma = 5, 
# colsample_bytree = 0.7, 
# min_child_weight = 2

ytrain <- as.numeric(factor(feat_6[tr,]))-1
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain[tr])
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))

xgb.grid.f6.1 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(8, 10, 12, 14),
      gamma = c(0, 3, 5),
      colsample_bytree = c(0.7, 0.85, 1.0),
      min_child_weight = c(0, 1, 2)
)

xgb.train.control.f6.1 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f6.1 <- train(x = alldata[1:1000,-1],
                       y=feat_6[1:1000,],
                       trControl = xgb.train.control.f6.1,
                       tuneGrid = xgb.grid.f6.1,
                       method = "xgbTree",
                       metric = "Accuracy"
                       
)



xgb.f6.1.max_depth <- xgb.prep.f6.1$bestTune$max_depth
xgb.f6.1.gamma <- xgb.prep.f6.1$bestTune$gamma
xgb.f6.1.colsample_bytree <- xgb.prep.f6.1$bestTune$colsample_bytree
xgb.f6.1.min_child_weight <- xgb.prep.f6.1$bestTune$min_child_weight

xgb.f6.1.params <- list(
      objective = "multi:softmax",
      num_class = 11,
      eta = 0.2,
      max_depth = xgb.f6.1.max_depth, 
      gamma = xgb.f6.1.gamma,
      colsample_bytree = xgb.f6.1.colsample_bytree,
      min_child_weight = xgb.f6.1.min_child_weight,
      metric = "Accuracy"
)

res.f6.1 <- xgb.cv(data = xtrain,
                   nrounds = 500,
                   nfold = 4,
                   params = xgb.f6.1.params,
                   early_stopping_rounds = 10,
                   print_every_n = 10
)

nrounds.f6.1 <- res.f6.1$best_iteration     

xgb.f6.1 <- xgboost(xtrain, params = xgb.f6.1.params,
                    nfold = 4, 
                    nrounds = nrounds.f6.1, 
                    early_stopping_rounds = 10)

pred.train.xgb.f6.1 <- predict(xgb.f6.1, xtrain)
pred.test.xgb.f6.1 <- predict(xgb.f6.1, xtest)

# table(pred.train.xgb.f6.1, feat_1[tr,])   pred.train.xgb.f6.1     X0     X1
# 0 116489  16705
# 1   7390  47734

xgb.grid.f6.2 <- expand.grid(
      nrounds = 10,
      eta = .2,
      max_depth = c(9, 10, 11, 12),
      gamma = c(2, 3, 4),
      colsample_bytree = c(0.7, 0.75, .8),
      min_child_weight = 1
)

xgb.train.control.f6.2 <- trainControl(
      method = "cv",
      number = 4,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      classProbs = TRUE,
      allowParallel = TRUE
)


xgb.prep.f6.2 <- train(x = alldata[1:1000,-1],
                    y=feat_6[1:1000,],
                    trControl = xgb.train.control.f6.2,
                    tuneGrid = xgb.grid.f6.2,
                    method = "xgbTree",
                    metric = "Accuracy"
                    
)



xgb.f6.2.max_depth <- xgb.prep.f6.2$bestTune$max_depth
xgb.f6.2.gamma <- xgb.prep.f6.2$bestTune$gamma
xgb.f6.2.colsample_bytree <- xgb.prep.f6.2$bestTune$colsample_bytree
xgb.f6.2.min_child_weight <- xgb.prep.f6.2$bestTune$min_child_weight

xgb.f6.2.params <- list(
      objective = "multi:softmax",
      num_class = 11,
      eta = .2,
      max_depth = xgb.f6.2.max_depth, 
      gamma = xgb.f6.2.gamma,
      colsample_bytree = xgb.f6.2.colsample_bytree,
      min_child_weight = xgb.f6.2.min_child_weight,
      metric = "Accuracy"
)


res.f6.2 <- xgb.cv(data = xtrain,
                nrounds = 1000,
                nfold = 4,
                params = xgb.f6.2.params,
                early_stopping_rounds = 10,
                print_every_n = 10
)

nrounds.f6.2 <- res.f6.2$best_iteration  

xgb.f6.2 <- xgboost(xtrain, nrounds = nrounds.f6.2,
                 params = xgb.f6.2.params,
                 early_stopping_rounds = 5,
                 nfold = 4
)

pred.train.xgb.f6.2 <- predict(xgb.f6.2, xtrain)
pred.test.xgb.f6.2 <- predict(xgb.f6.2, xtest)

# table(as.numeric(factor(feat_2$feat_2[tr])), pred)    ### no f.f3.1 in alldata ###
# pred
# 1     2     3
# 1 56393  8530  4326
# 2 18382 23540 11236
# 3  6135  7810 51966

# pred.train.xgb.f6.2
# 0     1     2
# 1 57983  9837  1429
# 2 19018 29161  4979
# 3  7157 10409 48345


# filtered
# pred.train.xgb.f6.2             X1    X2    X3
#                        0  66697 18449  9543
#                        1   1733 11409  1930
#                        2   6229 11183 57358


# filtered and adjusted around z.score = 0
# pred.train.xgb.f6.2       pred.train.xgb.f6.2
#                             0      1      2
#                             1 110234      0   6140
#                             2    822      7    454
#                             3  15101      0  51773

f.6 <- data.frame(f.6 = c(pred.train.xgb.f6.2, pred.test.xgb.f6.2))
save(f.6, file = "feature.6.RData")
save(xgb.f6.2_1, file = "xgb.f6.2.RData")
alldata <- cbind(alldata, f.6)


save(alldata, file = "df.RData")
save(tr, file = "tr.RData")
save(te, file = "te.RData")
save(training.loss, file = "training.loss.RData")
save(training.id, file = "training.id.RData")
save(test.id, file = "test.id.RData")
gc()
##############################  END  ###############################




