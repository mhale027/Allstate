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

feat.1[tr] <- ifelse(training.loss > mean.loss, 1, 0)


alldata$loss <- as.factor(feat.1)
names(alldata)[1] <- "feat.1"
levels(alldata$feat.1) <- make.names(levels(alldata$feat.1), unique = TRUE)


x.vars <- names(alldata)[-1]
response <- names(alldata)[1]

cat.vars <- names(alldata)[grep("cat", names(alldata))]
cont.vars <- names(alldata)[grep("cont", names(alldata))]

feat.2 <- 






h2o.init(nthreads = -1)

train <- as.h2o(alldata[tr,])
test <- as.h2o(alldata[te,-1])

sets <- h2o.splitFrame(train, .6)

train.hex <- h2o.assign(sets[[1]], "train.hex")
valid.hex <- h2o.assign(sets[[2]], "valid.hex")

dnn.1 <- h2o.deeplearning(x=x.vars,
                          y=response,
                          training_frame = train.hex,
                          validation_frame = valid.hex,
                          model_id = "dnn.1",
                          epochs = 5,
                          variable_importance = TRUE,
                          rate = .05,
                          rate_annealing = .001
)


pred.train.dnn.1 <- as.numeric(unlist(as.data.frame(predict(dnn.1, train))))
pred.test.dnn.1 <- as.numeric(unlist(as.data.frame(predict(dnn.1, test))))





dnn.2 <- h2o.deeplearning(x=x.vars,
                          y=response,
                          training_frame = train.hex,
                          validation_frame = valid.hex,
                          overwrite_with_best_model = FALSE,
                          hidden = c(200,50,50),
                          epochs = 10,
                          score_duty_cycle = .025,
                          adaptive_rate = FALSE,
                          rate = .01,
                          rate_annealing = 2e-6,
                          momentum_start = .2,
                          momentum_stable = .4,
                          momentum_ramp = 1e7,
                          l1 = 1e-5,
                          l2 = 1e-5,
                          max_w2 = 10
)



pred.train.dnn.2 <- as.numeric(unlist(as.data.frame(predict(dnn.2, train))))
pred.test.dnn.2 <- as.numeric(unlist(as.data.frame(predict(dnn.2, test))))

head(pred.dnn.2)
head(pred.test.dnn.2)

train.sample <- train.hex[1:10000, ]

hyper_params <- list(
      hidden = list(c(200, 100), c(150, 50), c(100, 35)),
      input_dropout_ratio = c(0, .05, .1),
      rate = c(.01, .005),
      rate_annealing = c(1e-7, 5e-6, 1e-6)
)

dnn.grid <- h2o.grid(
      algorithm = "deeplearning",
      grid_id = "dnn.grid",
      training_frame = train.sample,
      validation_frame = valid.hex,
      x = x.vars,
      y = response,
      epochs = 1,
      stopping_metric = "MSE",
      stopping_rounds = 2,
      score_validation_samples = 10000,
      score_duty_cycle = .025,
      # adaptive_rate = FALSE,
      # momentum_start = .5,
      # momentum_stable=0.9, 
      # momentum_ramp=1e7, 
      # l1=1e-5,
      # l2=1e-5,
      # activation = c("Rectifier"),
      # max_w2 = 10,
      hyper_params = hyper_params
)


dnn.grid <- data.frame(h2o.getGrid("dnn.grid")@summary_table)[1,]
























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



























































xgb.imp <- xgboost(data = xtrain, print_every_n = 10L, nrounds = 10)



var.imp <- xgb.importance(model = xgb.imp, feature_names = x.vars)
#cat_80_1 is #1 feature cont7, cat79_1, cont2, cat12_1


imp.vars <- var.imp$Feature
alldata1 <- alldata[,imp.vars]


for (k in grep("cat", names(alldata))) {
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
                  if ( "+" %in% strsplit(this.feat, split = "") == 0 ) {
                  } else {
                        alldata1[,1+ncol(alldata1)] <- (alldata1[,this.feat] + alldata1[,other.feats[i]])/2
                        names(alldata1)[ncol(alldata1)] <- paste0(this.feat, "+", other.feats[i])
                  }
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


xgb.1 <- xgboost(xtrain, 
                 print_every_n = 10L, 
                 nrounds = 100, 
                 # early_stopping_rounds = 5, 
                 base_score = 7.69
)

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

new.cols <- grep("+", names.alldata1)

alldata.feats <- cbind(alldata, alldata1[,new.cols])

xtrain <- xgb.DMatrix(data.matrix(alldata.feats[tr,-1]), label = ytrain)

xgb.2 <- xgboost(data = xtrain,
                 print_every_n = 10L, 
                 nrounds = 100,
                 base_score = 7.69
)

pred.train.xgb.2 <- exp(predict(xgb.2, xtrain))
pred.test.xgb.2 <- exp(predict(xgb.2, xtest))

xgb.params.3 <- list(objective = "reg:linear",
                     max_depth = 6,
                     eta = .1,
                     nfold = 5,
                     base_score = 7.69,
                     # min_child_weight = 1, 
                     subsample = .7, 
                     colsample_bytree = .7, 
                     # num_parallel_tree = 1,
                     eval_metric = "mae"
)

xgb.3 <- xgboost(data = xtrain,
                 print_every_n = 10L, nrounds = 500,
                 early_stopping_rounds = 10,
                 params = xgb.params.3
                 # booster = "gblinear",
                 # objective = "reg:linear"
)

pred.train.xgb.3 <- exp(predict(xgb.3, xtrain))
pred.test.xgb.3 <- exp(predict(xgb.3, xtest))

head(exp(training.loss))
head(pred.train.xgb.1)
head(pred.train.xgb.2)
head(pred.train.xgb.3)
head(pred.test.xgb.1)
head(pred.test.xgb.2)
head(pred.test.xgb.3)


############gbtree###############
# > head(exp(training.loss))
# [1] 2213.18 1283.60 3005.09  939.85 2763.85 5142.87
# > head(pred.train.xgb.1)
# [1] 2008.319 1703.520 4645.893 1027.234 3085.820 3853.907
# > head(pred.train.xgb.2)
# [1] 2026.745 1582.359 4197.879 1069.497 2982.383 4127.480
# > head(pred.train.xgb.3)
# [1] 2045.790 1699.724 3629.339 1248.771 3031.542 4233.112
# > head(pred.test.xgb.1)
# [1] 5012.134 8653.776 5015.730 1327.028 1424.418 6551.810
# > head(pred.test.xgb.2)
# [1] 1600.5289 2218.1783 5796.2343 5864.4857  856.8963 2510.0029
# > head(pred.test.xgb.3)
# [1] 1665.270 1975.686 4653.520 4969.422  997.977 2071.099
#################################

###########gblinear##############




rf <- ranger(guess~., alldata, num.trees = 50, write.forest = TRUE)





