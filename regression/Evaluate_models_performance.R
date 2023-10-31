.libPaths("/home/lars/RLibs/R4.2.1/")
require(ggplot2)
require(ggrepel)
require(glmnet)
require(randomForest)
require
source("~/cluster/project/SCG4SYN/read_in_data.R")
read_in_data("~/cluster/project/SCG4SYN/")
write.csv(libA, "output.csv", row.names=FALSE, quote=FALSE) 


predictions.felix <- read.csv("/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/10fold_cv/LibA_wide_pivot_state3_test_predicted_cv10fold_ensemble.csv")
predictions.felix <- ddply(predictions.felix, c("fold"), transform, predictions.std = scale(average_pred))
hist(predictions.felix$predictions.std)

libA$seq <- toupper(libA$seq)

state3e <- subset(libA, clusterID == "State_3E")
state3e <- merge(state3e, predictions.felix)
state3e <- ddply(state3e, c("fold"), transform, y.std = scale(mean.scaled))

# Random forest
rf <- readRDS("~/cluster/lvelten/Analysis/SCG4SYN/LibA_HSC/analysis/complete_run_simple_bio_analyses/007_predictions_RF.rds") 
qplot(x = norm.combined, y = predicted, data= rf)

# Deep Learning model
load("~/cluster/lvelten/Analysis/SCG4SYN/LibA_HSC/analysis/complete_run_simple_bio_analyses/007_R2_RF_byTF.rda")
qplot(x = mean.scaled, y = average_pred, data = state3e, color = factor(fold))

 # Calculate technical R**2
correlations <- state3e %>%
  group_by(TF) %>%
  summarize(correlation = cor(norm.1, norm.2)^2)

# Calculate correlation between predicted value and true value
R2_deep <- ddply(state3e , c("TF", "fold"), summarise, R2_deep_fold = cor(y.std, predictions.std)^2)
R2_deep <- ddply(R2_deep , "TF", summarise, R2_deep = mean(R2_deep_fold))
R2 <- merge(R2_deep, R2_RF_byTF)

# We normalize by the technical correlation
R2[, c(2,3,4)] <- R2[, c(2,3,4)]/correlations$correlation

# We plot the performance of the random forest vs the DL model
qplot(x = R2.grammar , y = R2_deep, data = R2) + geom_abline() + geom_text_repel(aes(label = TF),data = R2)

# we compute the mean of each of the folds
mean_per_fold <- ddply(state3e, "fold", summarise, measured = mean(y.std), predicted = mean(predictions.std))

