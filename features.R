# setwd("/users/matt/documents/projects/kaggle/allstate")
setwd("~/projects/kaggle/allstate")
training <- data.frame(fread("train.csv", header = TRUE))
test <- data.frame(fread("test.csv", header = TRUE))
sample <- data.frame(fread("sample_submission.csv", header = TRUE))

training$loss <- log(training$loss)

length.train <- nrow(training)
legnth.test <- nrow(test)

training.id <- training$id
test.id <- test$id


training.loss <- training$loss
mean.loss <- mean(training$loss)

training.length <- nrow(training)
test.length <- nrow(test)

tr <- 1:training.length
te <- (1+training.length):(training.length + test.length)

rm(training, test)


fold1 <- 1:35000
fold2 <- 35001:70000
fold3 <- 70001:105000
fold4 <- 105001:140000
fold5 <- 140001:188318

f1 <- tr[-fold1]
f2 <- tr[-fold2]
f3 <- tr[-fold3]
f4 <- tr[-fold4]
f5 <- tr[-fold5]