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
set.seed(7890)

load("data.RData")
load("features.RData")

sample <- data.frame(fread("sample_submission.csv", header = TRUE))

lengths <- features[[2]]
training.length <- lengths[1]
test.length <- lengths[2]

tr <- 1:training.length
te <- (1+training.length):(training.length + test.length)

ids <- features[[3]]
training.id <- ids[tr]
test.id <- ids[te]

losses <- features[[4]]
training.loss <- losses[tr]

x.vars <- names(alldata)[-1]
response <- names(alldata)[1]

cat.vars <- names(alldata)[grep("cat", names(alldata))]
cont.vars <- names(alldata)[grep("cont", names(alldata))]


mean.loss <- mean(training.loss)

ytrain <- data.matrix(alldata[tr,1])
xtrain <- xgb.DMatrix(data.matrix(alldata[tr,-1]), label = ytrain)
xtest <- xgb.DMatrix(data.matrix(alldata[te,-1]))
# test.base <- rep(7.69, test.length)
# setinfo(xtest, "base_margin", test.base)

# xgb.margin <- xgboost(xtrain, print_every_n = 10L, nrounds = 1, base_score = 7.69)
# ptrain <- predict(xgb.margin, xtrain, outputmargin = TRUE)
# setinfo(xtrain, "base_margin", ptrain)

# pytrain <- predict(xgb.margin, xtest, outputmargin = TRUE)
# setinfo(xtest, "base_margin", pytrain)


xgb.imp <- xgboost(data = xtrain, print_every_n = 10L, nrounds = 10)



var.imp <- xgb.importance(model = xgb.imp, feature_names = x.vars)
#cat_80_1 is #1 feature cont7, cat79_1, cont2, cat12_1


imp.vars <- var.imp$Feature
alldata1 <- alldata[,imp.vars]


for (k in 1:2) {
      
      cat.vars1 <- names(alldata1)[grep("cat", names(alldata1))]
      cont.vars1 <- names(alldata1)[grep("cont", names(alldata1))]
      this.feat <- imp.vars[k]
      
      if (this.feat %in% cat.vars1) {
           
            other.feats <- cat.vars1[-grep(this.feat, fixed = TRUE, cat.vars1)]
            
            for (i in 1:(length(cat.vars1)-1)) {
                  
                  if ( "+" %in% strsplit(this.feat, split = "") == 0 ) {
                  } else {
                  
                        alldata1[,1+ncol(alldata1)] <- alldata1[,this.feat] + alldata1[,other.feats[i]]
                  
                        names(alldata1)[ncol(alldata1)] <- paste0(this.feat, "+", other.feats[i])
                  }
            }
            
      } else {
            
            other.feats <- cont.vars1[-grep(this.feat, fixed = TRUE, cont.vars1)]
            
            for (i in 1:(length(cont.vars1)-1)) {
                  
                  alldata1[,1+ncol(alldata1)] <- (alldata1[,this.feat] + alldata1[,other.feats[i]])/2
                  names(alldata1)[ncol(alldata1)] <- paste(this.feat, "+", other.feats[i])
            }
      }
            
      xtrain <- xgb.DMatrix(data.matrix(alldata1[tr,]), label = ytrain)
      
      xgb.imp <- xgboost(data = xtrain, print_every_n = 10L, nrounds = 10)
      
      x.vars1 <- names(alldata1)
      
      var.imp2 <- xgb.importance(model = xgb.imp, feature_names = x.vars1)
      
      imp.vars <<- unique(var.imp2$Feature)
      
      alldata1 <- alldata1[,imp.vars]
      
}

xtrain <- xgb.DMatrix(data.matrix(alldata1[tr,]), label = ytrain)


xgb.1 <- xgboost(xtrain, print_every_n = 10L, nrounds = 100, early_stopping_rounds = 5, base_score = 7.69)
pred.train.xgb.1 <- exp(predict(xgb.1, xtrain))
pred.test.xgb.1 <- exp(predict(xgb.1, xtest))


########## ONLY SET MARGIN ON XTRAIN############
# exp(head(training.loss))
# [1] 2213.18 1283.60 3005.09  939.85 2763.85 5142.87
# > head(pred.train.xgb.1)
# [1] 2255.977 1538.311 4303.409 1013.668 2951.799 4493.379
# > head(pred.test.xgb.1)
# [1]  215.8424 2310.9264 4255.4309 1023.2449  355.1928  298.7702
################################################

########## SET MARGIN ON XTRAIN & XTEST#########
# exp(head(training.loss))
# [1] 2213.18 1283.60 3005.09  939.85 2763.85 5142.87
# > head(pred.test.xgb.1)
# [1]  226.1975 2421.7943 6271.3968 1602.8140  352.1719  313.1037
# > head(pred.train.xgb.1)
# [1] 2255.977 1538.311 4303.409 1013.668 2951.799 4493.379
################################################

########## ONLY SET MARGIN ON XTEST############
# exp(head(training.loss))
# [1] 2213.18 1283.60 3005.09  939.85 2763.85 5142.87
# > head(pred.train.xgb.1)
# [1] 2203.800 1529.774 4415.910 1102.628 2659.977 4523.226
# > head(pred.test.xgb.1)
# [1]  196.7147  487.7301 2025.3199 1717.5648  602.9352 2605.3843
# > 
################################################

names.alldata1 <- names(alldata1)

new.cols <- grep("+", names(alldata1))

alldata.feats <- cbind(alldata, alldata1[,new.cols])



xgb.2 <- xgboost(data = xgb.DMatrix(data.matrix(alldata.feats[tr,-1]), label = data.matrix(training.loss)), 
                 print_every_n = 10L, 
                 nrounds = 100,
                 base_score = 7.69)
pred.train.xgb.2 <- exp(predict(xgb.2, xgb.DMatrix(data.matrix(alldata.feats[tr,-1]), label = data.matrix(training.loss))))
pred.test.xgb.2 <- exp(predict(xgb.2, xgb.DMatrix(data.matrix(alldata.feats[te,-1]))))



xgb.params.3 <- list(objective = "reg:linear",
                     booster = "gbtree",
                     nfold = 3,
                     max_depth = 12,
                     eta = .01,
                     gamma = 2.5,
                     colsample_bytree = 1,
                     min_child_weight = 1.3,
                     base_score = 7.69
)
xgb.3 <- xgboost(data = xgb.DMatrix(data.matrix(alldata.feats[tr,-1]), label = data.matrix(training.loss)),
                 print_every_n = 10L, nrounds = 200,
                 early_stopping_rounds = 10,
                 params = xgb.params.3)

pred.train.xgb.3 <- exp(predict(xgb.3, data.matrix(alldata.feats[tr,-1])))
pred.test.xgb.3 <- exp(predict(xgb.3, data.matrix(alldata.feats[te,-1])))

head(exp(training.loss))
head(pred.train.xgb.1)
head(pred.train.xgb.2)
head(pred.train.xgb.3)
head(pred.test.xgb.1)
head(pred.test.xgb.2)
head(pred.test.xgb.3)











