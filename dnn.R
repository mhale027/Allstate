library(dplyr)
library(ggplot2)
library(caret)
library(Metrics)
library(h2o)
library(h2oEnsemble)

setwd("~/Projects/kaggle/Allstate")
load("features.RData")

deep <- function(output, train, test, epochs=10, rate=.01, rate_annealing=2e-6,hidden = c(200,200,200) ) {
      
      h2o.init(nthreads = -1)
      
      tr <- nrow(train)
      te <- nrow(test)
      
      response <- 1
      features <- 2:ncol(train)
      
      train <- as.h2o(train)
      test <- as.h2o(test)
      
      sets <- h2o.splitFrame(train, .7)
      
      train.hex <- h2o.assign(sets[[1]], "train.hex")
      valid.hex <- h2o.assign(sets[[2]], "valid.hex")
      
      
      dnn <- h2o.deeplearning(x=features,
                                y=response,
                                training_frame = train.hex,
                                validation_frame = valid.hex,
                                overwrite_with_best_model = FALSE,
                                hidden = c(200,50,50),
                                epochs = epochs,
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
      save(dnn, file = "dnn.RData")
      pred.train <- as.numeric(unlist(as.data.frame(predict(output, train))))
      pred.test <- as.numeric(unlist(as.data.frame(predict(output, test))))

      
      pred <- c(pred.train, pred.test)
}