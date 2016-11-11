load("train.raw.RData")
load("test.raw.RData")


length.train <- 188318
legnth.test <- 125546

training.id <- training$id
test.id <- test$id


training.loss <- training$loss
mean.loss <- mean(training$loss)

training.length <- nrow(training)
test.length <- nrow(test)

tr <- 1:training.length
te <- (1+training.length):(training.length + test.length)

rm(training, test)