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
feat.2 <- rep(0, nrow(alldata))
feat.3 <- rep(0, nrow(alldata))
feat.4 <- rep(0, nrow(alldata))
feat.5 <- rep(0, nrow(alldata))
feat.6 <- rep(0, nrow(alldata))


feat.1[tr] <- ifelse(training.loss > mean.loss, 1, 0)



z.score <- (training.loss - mean.loss)/sd(training.loss)
for (i in tr) {
      if (z.score[i] < -.5) {
            feat.2[i] <- 1
      } else if (z.score[i] < .02) {
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











alldata$loss <- as.factor(feat.1)
names(alldata)[1] <- "feat.1"
levels(alldata$feat.1) <- make.names(levels(alldata$feat.1), unique = TRUE)

cat.vars <- names(alldata)[grep("cat", names(alldata))]
cont.vars <- names(alldata)[grep("cont", names(alldata))]




