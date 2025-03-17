require(reshape2)
set.seed(1337)


load(url("https://figshare.com/ndownloader/files/53009321"))

rbind_overlaps <- function(...) {
  entries <- list(...)
  rbind_o_internal <- function(a,b){
    usecols <- intersect(colnames(a),colnames(b))
    rbind(a[,usecols],b[,usecols])
  }
  out <- rbind_o_internal(entries[[1]],entries[[2]])
  if (length(entries) > 2) {
    for (i in 3:length(entries)) {
      out <-rbind_o_internal(out,entries[[i]])
    }
  }
  return(out)
}


all_data <- rbind_overlaps(
  mpra.data$HSPC.libA$DATA,
  mpra.data$HSPC.libB$DATA,
  mpra.data$HSPC.libB$CONTROLS.GENERAL,
  mpra.data$HSPC.libB$CONTROLS.TP53,
  mpra.data$HSPC.libC$DATA,
  mpra.data$HSPC.libC$CONTROLS.GENERAL,
  mpra.data$HSPC.libC$CONTROLS.TP53,
  mpra.data$HSPC.libF$DATA,
  mpra.data$HSPC.libF$CONTROLS.GENERAL,
  mpra.data$HSPC.libF$CONTROLS.TP53
  )
all_data <- dcast(all_data, Seq + CRS + Library ~ clusterID, value.var = "mean.norm.adj")
all_data$Seq <- toupper(all_data$Seq)
all_data$Seq <- substr(all_data$Seq, nchar(all_data$Seq)-245, nchar(all_data$Seq))

write.csv(all_data, file = "data.all/complete_data.csv", quote=T, na = "nan", row.names=F)

set.seed(1337)
tenfold_cv <- sample(1:nrow(all_data))
all_data <- all_data[tenfold_cv, ]
folds <- rep(1:10, length.out = nrow(all_data))

for (i in 1:10) {
  j <- i + 1
  if (j == 11) j <- 1
  train <- all_data[folds != i & folds != j,]
  test <- all_data[folds == i,]
  valid <- all_data[folds == j,]
  write.csv(train, file = sprintf("data.all/all_train_%d.csv",i), quote=F, na = "nan", row.names=F)
  write.csv(valid, file = sprintf("data.all/all_valid_%d.csv",i), quote=F, na = "nan", row.names=F)
  write.csv(test, file = sprintf("data.all/all_test_%d.csv",i), quote=F, na = "nan", row.names=F)
  
  
}
