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

train <- alldata[tr, ]
test <- alldata[te, ]


h2o.init(nthreads = -1)

train <- as.h2o(train)
test <- as.h2o(test)

sets <- h2o.splitFrame(train, .6)

train.hex <- h2o.assign(sets[[1]], "train.hex")
valid.hex <- h2o.assign(sets[[2]], "valid.hex")

dnn.1 <- h2o.deeplearning(x=x.vars,
                          y=response,
                          training_frame = train.hex,
                          validation_frame = valid.hex,
                          model_id = "dnn.1",
                          epochs = 1,
                          variable_importance = TRUE
)
                          

pred.dnn.1 <- as.numeric(unlist(as.data.frame(predict(dnn.1, train))))
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


train.sample <- train.hex[1:10000, ]

hyper_params <- list(
      hidden = list(c(200, 200), c(150, 50), c(200, 100, 35)),
      input_dropout_ratio = c(0, .05),
      rate = c(.005, .01, .02),
      rate_annealing = c(1e-8, 1e-7, 1e-6)
)

dnn.grid <- h2o.grid(
      algorithm = "deeplearning",
      grid_id = "dnn.grid",
      training_frame = train.sample,
      validation_frame = valid.hex,
      x = x.vars,
      y = response,
      epochs = 10,
      stopping_metric = "MSE",
      stopping_rounds = 2,
      score_validation_samples = 10000,
      score_duty_cycle = .025,
      adaptive_rate = FALSE,
      momentum_start = .5,
      momentum_stable=0.9, 
      momentum_ramp=1e7, 
      l1=1e-5,
      l2=1e-5,
      activation = c("Rectifier"),
      max_w2 = 10,
      hyper_params = hyper_params
)

dnn.grid <- h2o.getGrid("dnn.grid")
















write.csv(sample,'submission.11.2.1.1.csv',row.names = FALSE)




h2o.shutdown()