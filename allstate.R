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



#create new vectors with the loss amounts and ids so they can be removed from the training set
training.id <- training$id
test.id <- test$id

training.loss <- training$loss

training <- select(training, -id, -loss)
test <- select(test, -id)

tr <- 1:nrow(training)
te <- (1 + nrow(training)):(nrow(test) + nrow(training))

test.loss <- rep(NA, nrow(test))

training <- cbind(loss = training.loss, training)
test <- cbind(test.loss, test)
names(training)[1] <- "loss"
names(test)[1] <- "loss"

alldata <- data.table(rbind(training, test))

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

alldata <- data.frame(alldata)
cols <- names(alldata)

for (f in cols) {
      if (class(alldata[[f]])=="character") {
            #cat("VARIABLE : ",f,"\n")
            levels <- unique(alldata[[f]])
            alldata[[f]] <- as.integer(factor(alldata[[f]], levels=levels))
      }
}

x.vars <- names(alldata)[-1]

# ytrain <- data.matrix(training.loss)
# xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain)
# xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))
# xgb.imp <- xgboost(data = xtrain, print_every_n = 10L, nrounds = 10)
# var.imp <- xgb.importance(model = xgb.imp, feature_names = x.vars)
# imp.vars <- var.imp$Feature
# alldata1 <- alldata[,imp.vars]
# alldata2 <- c(V1 = rep(NA, nrow(alldata)))
# count <- 0
# for (k in grep("cat", names(alldata)[1:20])) {
#       
#       count <- count + 1
#       
#       cat.vars1 <<- names(alldata1)[grep("cat", names(alldata1))]
#       cont.vars1 <<- names(alldata1)[grep("cont", names(alldata1))]
#       this.feat <<- imp.vars[count]
#       if (this.feat %in% cat.vars1) {
#             other.feats <<- cat.vars1[-grep(this.feat, fixed = TRUE, cat.vars1)]
#             for (i in 1:(length(cat.vars1)-1)) {
#                   if ( "+" %in% strsplit(this.feat, split = "") == FALSE ) {
#                         alldata2 <- alldata1[,this.feat] + alldata1[,other.feats[i]]
#                         # names(alldata2) <- paste0(this.feat, "+", other.feats[i])
#                   }
#             }
#       } else {
#             other.feats <<- cont.vars1[-grep(this.feat, fixed = TRUE, cont.vars1)]
#             for (i in 1:(length(cont.vars1)-1)) {
#                   if ( "+" %in% strsplit(this.feat, split = "") == FALSE ) {
#                         alldata2 <- (alldata1[,this.feat] + alldata1[,other.feats[i]])/2
#                         # names(alldata2)[1] <- paste0(this.feat, "+", other.feats[i])
#                   }
#             }
#       }
#       df1 <- data.frame(cbind(alldata1, alldata2))
#       names(df1)[81] <- this.feat
#       xtrain <- xgb.DMatrix(data.matrix(df[tr,]), label = ytrain)
#       xgb.imp <- xgboost(data = xtrain, print_every_n = 10L, nrounds = 10)
#       x.vars1 <- names(df)
#       var.imp2 <- xgb.importance(model = xgb.imp, feature_names = x.vars1)
#       imp.vars <- unique(var.imp2$Feature)
#       alldata1 <<- alldata1[,imp.vars]
# }
# 
# 
# 
# 
# 
# 
# 
# 
# 






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


vars = names(alldata)

for (f in vars) {
      if (class(alldata[[f]])=="character") {
            
            levels <- unique(alldata[[f]])
            alldata[[f]] <- as.integer(factor(alldata[[f]], levels=levels))
      }
}


# 
# 
# clean.set <- function(input) {
#       input <- as.data.frame(input)
#       n <- nrow(input)
#       for (i in grep("cat", names(input))) {
#             new.col <<- rep(NA, n)
#             choices <<- unique(input[,i])
#             for (k in 1:length(choices)) {
#                   new.col[input[,i] == choices[k]] <- k
#             }
#             input[,i] <- as.numeric(new.col)
#       }
#       
#       input <- input
# }
# 
# alldata <- clean.set(alldata)

load("template.RData")
load("nzv.t.RData")

load("df.RData")

ytrain <- data.matrix(training.loss)
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,2:141]), label = ytrain)
xtest <- xgb.DMatrix(data.matrix(alldata[te,2:141]))


params <- list(
      booster = "gbtree",
      objective = "reg:linear", 
      eta = 0.05, 
      subsample = .8, 
      colsample_bytree = 0.9, 
      min_child_weight = 1,
      # base_score = 7.69,
      num_parallel_tree = 1,
      metric = "ROC"
)

xg_eval_mae <- function (yhat, data) {
      labels = getinfo(data, "label")
      err= as.numeric(mean(abs(labels - yhat)))
      return (list(metric = "error", value = err))
}



xgb.grid <- expand.grid(
      nrounds = 10, 
      eta = 0.1, 
      max_depth = c(10, 12, 14), 
      gamma = c(0, 3, 5), 
      colsample_bytree = c(0.7, 0.85, 1.0), 
      min_child_weight = c(0, 1, 2)
)


xgb.train.control <- trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      allowParallel = TRUE
)




xgb.prep <- train(x=alldata[1:1000,-1],
                  y=alldata[1:1000,1],
                  trControl = xgb.train.control,
                  tuneGrid = xgb.grid,
                  method = "xgbTree"
)

# Fitting 
# nrounds = 10, 
# max_depth = 10, 
# eta = 0.1, 
# gamma = 5, 
# colsample_bytree = 0.7, 
# min_child_weight = 2


xgb.1.max_depth <- xgb.prep$bestTune$max_depth
xgb.1.gamma <- xgb.prep$bestTune$gamma
xgb.1.colsample_bytree <- xgb.prep$bestTune$colsample_bytree
xgb.1.min_child_weight <- xgb.prep$bestTune$min_child_weight

xgb.1.params <- list(
      objective = "reg:linear",
      eta = 0.1,              
      max_depth = xgb.1.max_depth, 
      gamma = xgb.1.gamma,
      colsample_bytree = xgb.1.colsample_bytree,
      min_child_weight = xgb.1.min_child_weight
      # base_score = 7.7
)

xgb.CV <- xgb.cv(data = xtrain,
                 params = xgb.1.params,
                 early_stopping_rounds = 10,
                 nfold = 4,
                 nrounds = 3000,
                 feval = xg_eval_mae,
                 maximize = FALSE,
                 print_every_n = 10
)


nrounds <- xgb.CV$best_iteration


xgb.1 <- xgboost(xtrain, params = xgb.1.params, 
                 nrounds = nrounds,
                 nfold = 4, 
                 early_stopping_rounds = 5)

pred.train.xgb.1 <- predict(xgb.1, xtrain)
pred.test.xgb.1 <- predict(xgb.1, xtest)

head(pred.train.xgb.1)
head(pred.test.xgb.1)
#[1]            1735.7131  2217.6819 10178.3232  6438.7598   838.8784  2768.6628      LB: 1165.80875
# with features 1635.2185  1993.2141 9452.0166   7079.5620   978.2036  1973.6853      LB: 1201.82510
# with f.1      1934.1953  1988.7725 8322.2188   5864.6196   869.9561  1965.6763

xgb.grid.2 <- expand.grid(
      nrounds = 10, 
      eta = 0.1, 
      max_depth = c(9, 10, 11), 
      gamma = c(4, 5, 6), 
      colsample_bytree = c(0.65, 0.7, .8), 
      min_child_weight = c(1.5, 2, 2.5)
)

xgb.train.control.2 <- trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      allowParallel = TRUE
)


xgb.prep.2 <- train(x = alldata[1:1000,-1],
                    y = alldata[1:1000,1],
                    trControl = xgb.train.control.2,
                    tuneGrid = xgb.grid.2,
                    method = "xgbTree"
)


# Fitting nrounds = 10, max_depth = 9, eta = 0.1, gamma = 4, colsample_bytree = 0.8, min_child_weight = 2.5 on full training set

xgb.2.max_depth <- xgb.prep.2$bestTune$max_depth
xgb.2.gamma <- xgb.prep.2$bestTune$gamma
xgb.2.colsample_bytree <- xgb.prep.2$bestTune$colsample_bytree
xgb.2.min_child_weight <- xgb.prep.2$bestTune$min_child_weight


xgb.2.params <- list(
      objective = "reg:linear",
      eta = 0.1,              
      max_depth = xgb.2.max_depth, 
      gamma = xgb.2.gamma,
      colsample_bytree = xgb.2.colsample_bytree,
      min_child_weight = xgb.2.min_child_weight
      # base_score = 7.7     
)


xgb.CV.2 <- xgb.cv(data = xtrain,
                 params = xgb.2.params,
                 early_stopping_rounds = 10,
                 nfold = 4,
                 nrounds = 3000,
                 feval = xg_eval_mae,
                 maximize = FALSE,
                 print_every_n = 10
)


nrounds.2 <- xgb.CV.2$best_iteration

xgb.2 <- xgboost(data = xtrain, early_stopping_rounds = 10, nrounds = nrounds.2, params = xgb.2.params)

pred.train.xgb.2 <- predict(xgb.2, xtrain)
pred.test.xgb.2 <- predict(xgb.2, xtest)

head(pred.train.xgb.2)
head(pred.test.xgb.2)
# [1]              1661.9288  2137.4812 10014.4863  5804.6362   914.8206  2579.1006  LB: 1167.78141
# with features    1689.3940  1901.8317 9412.9141   6829.9624   971.2341  2025.1431  LB: 1203.22653





xgb.grid.3 <- expand.grid(
      nrounds = 10, 
      max_depth = c(10, 14, 16), 
      eta = c(.05, .07),
      gamma = 4, 
      colsample_bytree = c(.8, .825), 
      min_child_weight = c(2.25, 2.5)
)

xgb.train.control.3 <- trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      allowParallel = TRUE
)



xgb.prep.3 <- train(x=alldata[1:1000,-1],
                    y=alldata[1:1000,1],
                    trControl = xgb.train.control.3,
                    tuneGrid = xgb.grid.3,
                    method = "xgbTree"
)
# Fitting nrounds = 10, max_depth = 12, eta = 0.05, gamma = 3.75, colsample_bytree = 0.775, min_child_weight = 2.75

xgb.3.max_depth <- xgb.prep.3$bestTune$max_depth
xgb.3.gamma <- xgb.prep.3$bestTune$gamma
xgb.3.colsample_bytree <- xgb.prep.3$bestTune$colsample_bytree
xgb.3.min_child_weight <- xgb.prep.3$bestTune$min_child_weight

xgb.3.params <- list(objective = "reg:linear",
                     booster = "gbtree",
                     nfold = 3,
                     max_depth = xgb.3.max_depth,
                     eta = .1,
                     gamma = xgb.3.gamma,
                     colsample_bytree = xgb.3.colsample_bytree,
                     min_child_weight = xgb.3.min_child_weight
)


xgb.CV.3 <- xgb.cv(data = xtrain,
                   params = xgb.3.params,
                   early_stopping_rounds = 10,
                   nfold = 4,
                   nrounds = 3000,
                   feval = xg_eval_mae,
                   maximize = FALSE,
                   print_every_n = 10
)

nrounds.3 <- xgb.CV.3$best_iteration

xgb.3.1 <- xgboost(data = xtrain, early_stopping_rounds = 10, nrounds = nrounds.3, params = xgb.3.params)

pred.test.xgb.3 <- predict(xgb.3.1, xtest)
head(pred.test.xgb.3)


#new cols           1790.5922  2277.3706 10140.8750  6496.3413   831.6128  2300.1387
# + sparse          1778.787   2168.226  9900.617    6967.435    868.582   2650.579   11/7       LB: 1166.81036
# -sprs + features  1955.4972  1481.7275 9783.0225   7675.4507   880.4752  2174.181   1000       LB: 1223.12381
# -sprs + features  1761.6776  1895.6948 9365.6973   6809.6709   985.4151  1963.3564  100        LB: 1203.25453

# ytrain <- c(ytrain, pred.test.xgb.3)
# xtrain <- xgb.DMatrix(data.matrix(alldata[,2:141]), label = ytrain)


xgb.grid.final <- expand.grid(
      nrounds = 10, 
      max_depth = c(10, 14, 16), 
      eta = c(.05, .07),
      gamma = 4, 
      colsample_bytree = c(.8, .825), 
      min_child_weight = c(2.25, 2.5)
)

xgb.train.control.final <- trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE,
      returnData = FALSE,
      returnResamp = "all",
      allowParallel = TRUE
)



xgb.prep.final <- train(x=alldata[1:1000,-1],
                    y=alldata[1:1000,1],
                    trControl = xgb.train.control.final,
                    tuneGrid = xgb.grid.final,
                    method = "xgbTree"
)
# Fitting nrounds = 10, max_depth = 12, eta = 0.05, gamma = 3.75, colsample_bytree = 0.775, min_child_weight = 2.75

xgb.final.max_depth <- xgb.prep.final$bestTune$max_depth
xgb.final.gamma <- xgb.prep.final$bestTune$gamma
xgb.final.colsample_bytree <- xgb.prep.final$bestTune$colsample_bytree
xgb.final.min_child_weight <- xgb.prep.final$bestTune$min_child_weight

xgb.final.params <- list(objective = "reg:linear",
                     booster = "gbtree",
                     nfold = 3,
                     max_depth = xgb.final.max_depth,
                     eta = .1,
                     gamma = xgb.final.gamma,
                     colsample_bytree = xgb.final.colsample_bytree,
                     min_child_weight = xgb.final.min_child_weight
)


xgb.CV.final <- xgb.cv(data = xtrain,
                   params = xgb.final.params,
                   early_stopping_rounds = 10,
                   nfold = 4,
                   nrounds = 3000,
                   feval = xg_eval_mae,
                   maximize = FALSE,
                   print_every_n = 10
)

nrounds.final <- xgb.CV.final$best_iteration



xgb.final <- xgboost(data = xtrain, early_stopping_rounds = 10, nrounds = nrounds.final, params = xgb.final.params)

pred.test.xgb.final <- predict(xgb.final, xtest)
head(pred.test.xgb.final)



# 1764.109 1824.722 9534.673 6776.678  989.458 2018.350               LB: 1203.52243












X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb1 <- xgboost(data = X_train, params = params, max_depth = 1, nrounds = nrounds)
pred1 <- predict(xgb1, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred1 <- pred1

pred.t.1 <- predict(xgb1, xgb.DMatrix(data.matrix(test.set)))
test.set$pred1 <- pred.t.1




X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb2 <- xgboost(data = X_train, params = params, max_depth = 2, nrounds = nrounds)
pred2 <- pred2 <- predict(xgb2, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred2 <- pred2

pred.t.2 <- predict(xgb2, xgb.DMatrix(data.matrix(test.set)))
test.set$pred2 <- pred.t.2



X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb3 <- xgboost(data = X_train, params = params, max_depth = 3, nrounds = nrounds)
pred3 <- pred3 <- predict(xgb3, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred3 <- pred3

X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb4 <- xgboost(data = X_train, params = params, max_depth = 4, nrounds = nrounds)
pred4 <- pred4 <- predict(xgb4, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred4 <- pred4

X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb5 <- xgboost(data = X_train, params = params, max_depth = 5, nrounds = nrounds)
pred5 <- pred5 <- predict(xgb5, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred5 <- pred5

X_train <- xgb.DMatrix(data = data.matrix(training.set), label = labels)
xgb6 <- xgboost(data = X_train, params = params, max_depth = 6, nrounds = nrounds)
pred6 <- pred6 <- predict(xgb6, xgb.DMatrix(data.matrix(training.set), label = training.loss))
training.set$pred6 <- pred6


training.rf.3 <- training.rf
training.rf.3$pred.rf <- training.rf$pred.rf
training.rf.3$pred1 <- training.set$pred1
training.rf.3$pred2 <- training.set$pred2
training.rf.3$pred3 <- training.set$pred3
training.rf.3$pred4 <- training.set$pred4
training.rf.3$pred5 <- training.set$pred5
training.rf.3$pred6 <- training.set$pred6


test.pred1 <- predict(xgb1, xgb.DMatrix(data.matrix(test.set)))
test.pred2 <- predict(xgb2, xgb.DMatrix(data.matrix(test.set)))
test.pred3 <- predict(xgb3, xgb.DMatrix(data.matrix(test.set)))
test.pred4 <- predict(xgb4, xgb.DMatrix(data.matrix(test.set)))
test.pred5 <- predict(xgb5, xgb.DMatrix(data.matrix(test.set)))
test.pred6 <- predict(xgb6, xgb.DMatrix(data.matrix(test.set)))







training.rf <- training.set[1:1000,]
rf <- randomForest(y=training.loss[1:1000], x=training.rf[1:1000,], importance = TRUE, ntree = 5)
imp <- varImp(rf)
imp$var <- rownames(imp)
imp <- c(arrange(imp, desc(Overall))$var[1:100])
training.rf <- training.set[,imp]
rf.2 <- randomForest(y=training.loss, x=training.rf, ntree = 5)

pred.train.rf <- predict(rf.2, training.rf)
training.set$pred.rf <- pred.train.rf

validate.rf <- validate.set[,imp]
pred.validate.rf <- predict(rf.2, validate.rf)
validate.set$pred.rf <- pred.validate.rf




test.rf <- test.set[,imp]
pred.test.rf <- predict(rf.2, test.rf)
test.set$pred.rf <- pred.test.rf






test.rf.3 <- test.rf
test.rf.3$pred.rf <- test.rf$pred.rf
test.rf.3$pred1 <- test.pred1
test.rf.3$pred2 <- test.pred2
test.rf.3$pred3 <- test.pred3
test.rf.3$pred4 <- test.pred4
test.rf.3$pred5 <- test.pred5
test.rf.3$pred6 <- test.pred6


rf.3 <- randomForest(y=training.loss, x=training.rf, ntree = 50)
pred.train.rf.3 <- predict(rf.3, training.rf.3)
training.set$pred.rf.3 <- pred.train.rf.3

pred.test.rf.3 <- predict(rf.3, test.rf.3)
test.rf.3$pred.rf.3 <- pred.test.rf.3

sample$loss <- pred.test.rf.3

write.csv(sample,'submission.10.29.csv',row.names = FALSE)

#id     loss
#1  4 1685.679
#2  6 1834.864
#3  9 7124.580
#4 12 5947.825
#5 15 1617.754
#6 17 2762.761

# 
# id     loss
# 1  4 1654.149
# 2  6 1836.539
# 3  9 7772.634
# 4 12 5345.297
# 5 15 1420.574
# 6 17 2714.712
# 
# validated set
# 
# 
# # 


# gblinear
# id     loss
# 1  4 1602.187
# 2  6 1664.112
# 3  9 6229.759
# 4 12 4296.716
# 5 15 1371.976
# 6 17 2180.190