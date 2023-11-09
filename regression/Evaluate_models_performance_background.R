.libPaths("/home/lars/RLibs/R4.2.1/")
require(ggplot2)
require(ggrepel)
require(glmnet)
require(randomForest)
source("~/cluster/project/SCG4SYN/read_in_data.R")
read_in_data("~/cluster/project/SCG4SYN/")
write.csv(libA, "output.csv", row.names=FALSE, quote=FALSE) 

# Background predictions
predictions.background <- read.csv("/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/10fold_cv/LibA_wide_pivot_state3_test_predicted_cv10fold_background_ensemble.csv")
predictions.background$average_pred <- rowMeans(predictions.background[,c(5:14)])
predictions.background <- ddply(predictions.background, c("fold"), transform, predictions.std = scale(average_pred))
hist(predictions.background$predictions.std)

state3e_background <- subset(libA, clusterID == "State_3E")
state3e_background <- merge(state3e_background, predictions.background)
state3e_background <- ddply(state3e_background, c("fold"), transform, y.std = scale(mean.scaled))

# Deep Learning model trained on background
qplot(x = mean.scaled, y = average_pred, data = state3e_background, color = factor(fold))

# Calculate technical R**2
correlations_background <- state3e_background %>%
  group_by(TF) %>%
  summarize(correlation = cor(norm.1, norm.2)^2)

# Calculate correlation between predicted value and true value
R2_deep_background <- ddply(state3e_background , c("TF", "fold"), summarise, R2_deep_fold = cor(y.std, predictions.std)^2)
R2_deep_background <- ddply(R2_deep_background , "TF", summarise, R2_deep_background = mean(R2_deep_fold))

#####

# Calculate R2 for deep model
predictions.whole <- read.csv("/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/10fold_cv/LibA_wide_pivot_state3_test_predicted_cv10fold_ensemble.csv")
predictions.whole$average_pred <- rowMeans(predictions.whole[,c(5:13)])
predictions.whole <- ddply(predictions.whole, c("fold"), transform, predictions.std = scale(average_pred))
hist(predictions.whole$predictions.std)

libA$seq <- toupper(libA$seq)


state3e_whole <- subset(libA, clusterID == "State_3E")
state3e_whole <- merge(state3e_whole, predictions.whole)
state3e_whole <- ddply(state3e_whole, c("fold"), transform, y.std = scale(mean.scaled))

# Calculate technical R**2
correlations_whole <- state3e_whole %>%
  group_by(TF) %>%
  summarize(correlation = cor(norm.1, norm.2)^2)

# Calculate correlation between predicted value and true value
R2_deep_whole <- ddply(state3e_whole , c("TF", "fold"), summarise, R2_deep_fold = cor(y.std, predictions.std)^2)
R2_deep_whole<- ddply(R2_deep_whole , "TF", summarise, R2_deep = mean(R2_deep_fold))
R2_deep_whole
R2 <- merge(R2_deep_background, R2_deep_whole)


# We normalize by the technical correlation
R2[, c(2,3)] <- R2[, c(2,3)]/correlations_whole$correlation

# We plot the performance of the random forest vs the DL model
qplot(x = R2_deep_background , y = R2_deep, data = R2) + geom_abline() + geom_text_repel(aes(label = TF),data = R2)
