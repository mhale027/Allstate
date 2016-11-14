setwd("/users/matt/documents/projects/kaggle/allstate")
training <- data.frame(fread("train.csv", header = TRUE))
test <- data.frame(fread("test.csv", header = TRUE))
sample <- data.frame(fread("sample_submission.csv", header = TRUE))

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