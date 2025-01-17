.libPaths("/home/lars/RLibs/R4.2.1/")
require(ggplot2)
require(ggrepel)
require(glmnet)
require(randomForest)
source("~/cluster/project/SCG4SYN/read_in_data.R")
read_in_data("~/cluster/project/SCG4SYN/")

# Load library A 
libA <- DATA$HSPC.libA.BS1$narrow
colnames(libA)[15] <- "seq"
colnames(libA)[27] <- "mean.scaled"
libA_subset <- subset(libA, clusterID == "State_3E")



###
### BACKGROUND
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

# Calculate correlation between predicted value and true value
R2_deep_background <- ddply(state3e_background , c("TF", "fold"), summarise, R2_deep_fold = cor(y.std, predictions.std)^2)
R2_deep_background <- ddply(R2_deep_background , "TF", summarise, R2_deep_background = mean(R2_deep_fold))

####
#### ONLY MOTIFS
predictions.motifs <- read.csv("/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/10fold_cv/LibA_wide_pivot_state3_test_predicted_cv10fold_motifs_only_ensemble.csv")
predictions.motifs$average_pred <- rowMeans(predictions.motifs[,c(4:12)])
predictions.motifs <- ddply(predictions.motifs, c("fold"), transform, predictions.std = scale(average_pred))
hist(predictions.motifs$predictions.std)

state3e.motifs <- subset(libA, clusterID == "State_3E")
state3e.motifs <- merge(state3e.motifs, predictions.motifs)
state3e.motifs <- ddply(state3e.motifs, c("fold"), transform, y.std = scale(mean.scaled))

# Calculate correlation between predicted value and true value
R2_deep_motifs <- ddply(state3e.motifs , c("TF", "fold"), summarise, R2_deep_motifs_fold = cor(y.std, predictions.std)^2)
R2_deep_motifs <- ddply(R2_deep_motifs , "TF", summarise, R2_deep_motifs = mean(R2_deep_motifs_fold))

###
### WHOLE MODEL
# Calculate R2 for deep model
predictions.whole <- read.csv("/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/10fold_cv/LibA_wide_pivot_state3_test_predicted_cv10fold_ensemble.csv")
predictions.whole$average_pred <- rowMeans(predictions.whole[,c(5:13)])
predictions.whole <- ddply(predictions.whole, c("fold"), transform, predictions.std = scale(average_pred))
hist(predictions.whole$predictions.std)
libA$seq <- toupper(libA$seq)
state3e_whole <- subset(libA, clusterID == "State_3E")
state3e_whole <- merge(state3e_whole, predictions.whole)
state3e_whole <- ddply(state3e_whole, c("fold"), transform, y.std = scale(mean.scaled))

# Calculate correlation between predicted value and true value
R2_deep_whole <- ddply(state3e_whole , c("TF", "fold"), summarise, R2_deep_fold = cor(y.std, predictions.std)^2)
R2_deep_whole<- ddply(R2_deep_whole , "TF", summarise, R2_deep = mean(R2_deep_fold))
R2_deep_whole

# Calculate technical R**2
correlations <- state3e.motifs %>%
  group_by(TF) %>%
  summarize(correlation = cor(norm.1.adj, norm.2.adj)^2)

# We normalize by the technical correlation
load("~/cluster/lvelten/Analysis/SCG4SYN/LibA_HSC/analysis/complete_run_simple_bio_analyses/007_R2_RF_byTF.rda")

R2 <- merge(R2_deep_whole, R2_RF_byTF)
R2 <- merge(R2, R2_deep_background)
R2 <- merge(R2, R2_deep_motifs)
R2[, c(2,3,4,5,6)] <- R2[, c(2,3,4,5,6)]/correlations$correlation

# We plot the performance of the random forest vs the DL model
qplot(x = R2_deep_background , y = R2_deep, data = R2) + geom_abline() + geom_text_repel(aes(label = TF),data = R2)
qplot(x = R2_deep_motifs , y = R2_deep, data = R2) + geom_abline() + geom_text_repel(aes(label = TF),data = R2)
qplot(x = R2_deep_motifs , y = R2_deep_background, data = R2) + geom_abline() + geom_text_repel(aes(label = TF),data = R2)


