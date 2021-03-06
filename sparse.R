
sparse <- function(train) {
      train <- data.table(train)
      c.vars <- names(train)[grep("cat", names(train))]
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

