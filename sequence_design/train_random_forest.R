#look at design replicates in lib H, and other analyses that look at R2
#.libPaths("/home/lars/RLibs/R4.2.1/")
require(ggplot2)
require(ggrepel)
require(glmnet)
require(ranger)
require(reshape2)
require(parallel)


load(url("https://figshare.com/ndownloader/files/53009321"))

#try narrow format first
libH.forRF <- subset(mpra.data$HSPC.libB$DATA, select = c("CRS", "Seq","clusterID","mean.scaled.final","TFnumber", "TForder","TF1.name", "TF1.affinity", "TF1.orientation", "TF2.name", "TF2.affinity", "TF2.orientation", "spacer"))
new.TF1 <- with(libH.forRF, ifelse(TF1.name > TF2.name, TF1.name,TF2.name))
new.TF2 <- with(libH.forRF, ifelse(TF1.name > TF2.name, TF2.name,TF1.name))
new.TF1.aff <- with(libH.forRF, ifelse(TF1.name > TF2.name, TF1.affinity,TF2.affinity))
new.TF2.aff <- with(libH.forRF, ifelse(TF1.name > TF2.name, TF2.affinity,TF1.affinity))
new.TF1.ori <- with(libH.forRF, ifelse(TF1.name > TF2.name, TF1.orientation,TF2.orientation))
new.TF2.ori <- with(libH.forRF, ifelse(TF1.name > TF2.name, TF2.orientation,TF1.orientation))
libH.forRF$TF1.name <- new.TF1
libH.forRF$TF2.name <- new.TF2
libH.forRF$TF1.affinity <- new.TF1.aff
libH.forRF$TF2.affinity <- new.TF2.aff
libH.forRF$TF1.orientation <- new.TF1.ori
libH.forRF$TF2.orientation <- new.TF2.ori
libH.forRF$TFnumber <- 2*libH.forRF$TFnumber

libA.forRF <- subset(mpra.data$HSPC.libA$DATA, select = c("CRS", "Seq","clusterID", "mean.scaled.final"))
libA.forRF$TFnumber <- mpra.data$HSPC.libA$DATA$nrepeats
libA.forRF$TForder <- "Block"
libA.forRF$TF1.name <- mpra.data$HSPC.libA$DATA$TF
libA.forRF$TF2.name <- mpra.data$HSPC.libA$DATA$TF
libA.forRF$TF1.affinity <- mpra.data$HSPC.libA$DATA$affinitynum
libA.forRF$TF2.affinity <- mpra.data$HSPC.libA$DATA$affinitynum
libA.forRF$TF1.orientation <- with(mpra.data$HSPC.libA$DATA , ifelse(orientation=="tandem","fwd",orientation))
libA.forRF$TF2.orientation <- with(mpra.data$HSPC.libA$DATA , ifelse(orientation=="tandem","rev",orientation))
libA.forRF$spacer <-mpra.data$HSPC.libA$DATA

libA.forRF$Seq <- substr(libA.forRF$Seq, 17,262)
forRF <- rbind(libA.forRF, libH.forRF)

folds <- readRDS("seq2fold.rds")
colnames(folds)[1] <- "Seq"
forRF <- merge(forRF, folds)

    tenfold_reg <- do.call(rbind, lapply(1:10, function(i) {
      train <- subset(forRF, fold != i)
      test <- subset(forRF, fold == i)
      model.grammar <- ranger(mean.scaled.final ~ TFnumber + TF1.name + TF2.name + TF1.affinity + TF2.affinity + spacer + TForder + TF1.orientation + TF2.orientation + clusterID ,data = train)
      #no benefit from adding interaction terms?
      #model.grammar <- randomForest(mean.scaled.final ~ (TFnumber * TF1.affinity * TF2.affinity) : spacer : TForder : TF1.orientation : TF2.orientation : clusterID + clusterID + TF1.affinity + TF2.affinity + TFnumber ,data = train)
      predicted <- predict(model.grammar, data = test)
      data.frame(test, predicted = predicted$predictions)
    }))

#create two data frames
# a) Impute all NAs, keep available measurements
measured.wide <- dcast(tenfold_reg, CRS + fold + TFnumber + TForder + TF1.name + TF2.name + TF1.affinity + TF2.affinity + TF1.orientation + TF2.orientation + spacer ~ clusterID, value.var = "mean.scaled.final")
all.designs <- subset(measured.wide, select = 1:11)
all.designs <- merge(all.designs, data.frame(clusterID = unique(forRF$clusterID)))
all.designs$predicted <- NA
for ( i in 1:10) {
  all.designs$predicted[all.designs$fold==i] <- predict(models[[i]], data = subset(all.designs, fold == i))$predictions
}

# b) predict all measurements
predicted.wide <- dcast(all.designs, CRS + fold + TFnumber + TForder + TF1.name + TF2.name + TF1.affinity + TF2.affinity + TF1.orientation + TF2.orientation + spacer ~ clusterID, value.var = "predicted")

imputed.wide <- measured.wide
imputed.wide[,12:18][is.na(imputed.wide[,12:18])] <- predicted.wide[,12:18][is.na(imputed.wide[,12:18])]

save(imputed.wide, predicted.wide, file = "RF_results.rda")
