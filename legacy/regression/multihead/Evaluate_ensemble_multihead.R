.libPaths("/home/lars/RLibs/R4.2.1/")
require(ggplot2)
require(ggrepel)
require(glmnet)
require(randomForest)
source("~/cluster/project/SCG4SYN/read_in_data.R")
read_in_data("~/cluster/project/SCG4SYN/")

libA <- DATA$HSPC.libA.BS1$narrow
predictions.felix <- read.csv("/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/10fold_cv/ensemble_multihead_LibA_wide_pivot_state3_test_predicted_cv10fold.csv")

# Scale data 
columns_to_scale <- grep("^avg_prediction", names(predictions.felix), value = TRUE)
# Overwrite the original columns with scaled values
predictions.felix <- predictions.felix %>%
  mutate_at(vars(columns_to_scale), funs(scale(.)))

columns_to_scale <- grep("^State", names(predictions.felix), value = TRUE)
# Overwrite the original columns with scaled values
predictions.felix <- predictions.felix %>%
  mutate_at(vars(columns_to_scale), funs(scale(.)))

# Get column names
avg_prediction_columns <- grep("^avg_prediction", names(predictions.felix), value = TRUE)
state_columns <- c("State_1M", "State_2D", "State_3E", "State_4M" ,"State_5M", "State_6N", "State_7M")

library(gridExtra)

# Create an empty list to store the plots
plots_list <- list()

# Create scatter plots for each combination
for (i in 1:7) {
  # Extract column names
  avg_col <- avg_prediction_columns[i]
  state_col <- state_columns[i]
  
  # Calculate the correlation and R-squared
  corr <- cor(predictions.felix[[avg_col]], predictions.felix[[state_col]], use="complete.obs")^2
  
  # Create scatter plot
  scatter_plot <- ggplot(predictions.felix, aes(x = .data[[avg_col]], y = .data[[state_col]])) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    annotate("text", x = 6, y =8, 
             label = paste("R^2 =", round(corr, 3)), hjust = 1, vjust = 1) +    ggtitle(state_col)
  scatter_plot
  # Add scatter plot to the list
  plots_list[[length(plots_list) + 1]] <- scatter_plot
}

# Arrange the plots in a grid
grid.arrange(grobs = plots_list, ncol = length(state_columns))


# BACKGROUND

read_in_data("~/cluster/project/SCG4SYN/")
libA <- DATA$HSPC.libA.BS1$narrow
predictions.felix <- read.csv("/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/10fold_cv/ensemble_multihead_background_rndomdna_LibA_wide_pivot_state3_test_predicted_cv10fold.csv")

# Scale data 
columns_to_scale <- grep("^avg_prediction", names(predictions.felix), value = TRUE)
# Overwrite the original columns with scaled values
predictions.felix <- predictions.felix %>%
  mutate_at(vars(columns_to_scale), funs(scale(.)))

columns_to_scale <- grep("^State", names(predictions.felix), value = TRUE)
# Overwrite the original columns with scaled values
predictions.felix <- predictions.felix %>%
  mutate_at(vars(columns_to_scale), funs(scale(.)))

# Get column names
avg_prediction_columns <- grep("^avg_prediction", names(predictions.felix), value = TRUE)
state_columns <- c("State_1M", "State_2D", "State_3E", "State_4M" ,"State_5M", "State_6N", "State_7M")


# Create an empty list to store the plots
plots_list <- list()

# Create scatter plots for each combination
for (i in 1:7) {
  # Extract column names
  avg_col <- avg_prediction_columns[i]
  state_col <- state_columns[i]
  
  # Calculate the correlation and R-squared
  corr <- cor(predictions.felix[[avg_col]], predictions.felix[[state_col]], use="complete.obs")^2
  
  # Create scatter plot
  scatter_plot <- ggplot(predictions.felix, aes(x = .data[[avg_col]], y = .data[[state_col]])) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    annotate("text", x = 6, y =8, 
             label = paste("R^2 =", round(corr, 3)), hjust = 1, vjust = 1) +    ggtitle(state_col)
  scatter_plot
  # Add scatter plot to the list
  plots_list[[length(plots_list) + 1]] <- scatter_plot
}

# Arrange the plots in a grid
grid.arrange(grobs = plots_list, ncol = length(state_columns))


libA <- DATA$HSPC.libA.BS1$narrow
predictions.felix <- read.csv("/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/10fold_cv/ensemble_multihead_LibA_wide_pivot_state3_test_predicted_cv10fold.csv")

# Scale data 
columns_to_scale <- grep("^avg_prediction", names(predictions.felix), value = TRUE)
# Overwrite the original columns with scaled values
predictions.felix <- predictions.felix %>%
  mutate_at(vars(columns_to_scale), funs(scale(.)))

columns_to_scale <- grep("^State", names(predictions.felix), value = TRUE)
# Overwrite the original columns with scaled values
predictions.felix <- predictions.felix %>%
  mutate_at(vars(columns_to_scale), funs(scale(.)))

# Get column names
avg_prediction_columns <- grep("^avg_prediction", names(predictions.felix), value = TRUE)
state_columns <- c("State_1M", "State_2D", "State_3E", "State_4M" ,"State_5M", "State_6N", "State_7M")

library(gridExtra)

# Create an empty list to store the plots
plots_list <- list()

# Create scatter plots for each combination
for (i in 1:7) {
  # Extract column names
  avg_col <- avg_prediction_columns[i]
  state_col <- state_columns[i]
  
  # Calculate the correlation and R-squared
  corr <- cor(predictions.felix[[avg_col]], predictions.felix[[state_col]], use="complete.obs")^2
  
  # Create scatter plot
  scatter_plot <- ggplot(predictions.felix, aes(x = .data[[avg_col]], y = .data[[state_col]])) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    annotate("text", x = 6, y =8, 
             label = paste("R^2 =", round(corr, 3)), hjust = 1, vjust = 1) +    ggtitle(state_col)
  scatter_plot
  # Add scatter plot to the list
  plots_list[[length(plots_list) + 1]] <- scatter_plot
}

# Arrange the plots in a grid
grid.arrange(grobs = plots_list, ncol = length(state_columns))


# MOTIFS only

read_in_data("~/cluster/project/SCG4SYN/")
libA <- DATA$HSPC.libA.BS1$narrow
predictions.felix <- read.csv("/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/10fold_cv/ensemble_multihead_motifs_LibA_wide_pivot_state3_test_predicted_cv10fold.csv")

# Scale data 
columns_to_scale <- grep("^avg_prediction", names(predictions.felix), value = TRUE)
# Overwrite the original columns with scaled values
predictions.felix <- predictions.felix %>%
  mutate_at(vars(columns_to_scale), funs(scale(.)))

columns_to_scale <- grep("^State", names(predictions.felix), value = TRUE)
# Overwrite the original columns with scaled values
predictions.felix <- predictions.felix %>%
  mutate_at(vars(columns_to_scale), funs(scale(.)))

# Get column names
avg_prediction_columns <- grep("^avg_prediction", names(predictions.felix), value = TRUE)
state_columns <- c("State_1M", "State_2D", "State_3E", "State_4M" ,"State_5M", "State_6N", "State_7M")


# Create an empty list to store the plots
plots_list <- list()

# Create scatter plots for each combination
for (i in 1:7) {
  # Extract column names
  avg_col <- avg_prediction_columns[i]
  state_col <- state_columns[i]
  
  # Calculate the correlation and R-squared
  corr <- cor(predictions.felix[[avg_col]], predictions.felix[[state_col]], use="complete.obs")^2
  
  # Create scatter plot
  scatter_plot <- ggplot(predictions.felix, aes(x = .data[[avg_col]], y = .data[[state_col]])) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    annotate("text", x = 6, y =8, 
             label = paste("R^2 =", round(corr, 3)), hjust = 1, vjust = 1) +    ggtitle(state_col)
  scatter_plot
  # Add scatter plot to the list
  plots_list[[length(plots_list) + 1]] <- scatter_plot
}

# Arrange the plots in a grid
grid.arrange(grobs = plots_list, ncol = length(state_columns))


# MOTIFS only + 2bp extension

read_in_data("~/cluster/project/SCG4SYN/")
libA <- DATA$HSPC.libA.BS1$narrow
predictions.felix <- read.csv("/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/10fold_cv/ensemble_multihead_motifs_2bp_extended_LibA_wide_pivot_state3_test_predicted_cv10fold.csv")

# Scale data 
columns_to_scale <- grep("^avg_prediction", names(predictions.felix), value = TRUE)
# Overwrite the original columns with scaled values
predictions.felix <- predictions.felix %>%
  mutate_at(vars(columns_to_scale), funs(scale(.)))

columns_to_scale <- grep("^State", names(predictions.felix), value = TRUE)
# Overwrite the original columns with scaled values
predictions.felix <- predictions.felix %>%
  mutate_at(vars(columns_to_scale), funs(scale(.)))

# Get column names
avg_prediction_columns <- grep("^avg_prediction", names(predictions.felix), value = TRUE)
state_columns <- c("State_1M", "State_2D", "State_3E", "State_4M" ,"State_5M", "State_6N", "State_7M")


# Create an empty list to store the plots
plots_list <- list()

# Create scatter plots for each combination
for (i in 1:7) {
  # Extract column names
  avg_col <- avg_prediction_columns[i]
  state_col <- state_columns[i]
  
  # Calculate the correlation and R-squared
  corr <- cor(predictions.felix[[avg_col]], predictions.felix[[state_col]], use="complete.obs")^2
  
  # Create scatter plot
  scatter_plot <- ggplot(predictions.felix, aes(x = .data[[avg_col]], y = .data[[state_col]])) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    annotate("text", x = 6, y =8, 
             label = paste("R^2 =", round(corr, 3)), hjust = 1, vjust = 1) +    ggtitle(state_col)
  scatter_plot
  # Add scatter plot to the list
  plots_list[[length(plots_list) + 1]] <- scatter_plot
}

# Arrange the plots in a grid
grid.arrange(grobs = plots_list, ncol = length(state_columns))





