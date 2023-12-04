# This script reads Library A data and makes different dataframes according to the modeling needs (with background, without, etc)

# Import libraries and data
library(tidyr)
source("~/cluster/project/SCG4SYN/read_in_data.R")
read_in_data("~/cluster/project/SCG4SYN")

# We subset the object to only library A
libA <- DATA$HSPC.libA.BS1$narrow
colnames(libA)[15] <- "seq"


# SIMPLE REGRESSION : Save state3E with upper case and IDS (for model that sees the whole sequence)
df_libraryA <- libA[, c("clusterID", "seq", "mean.scaled.tp53","CRS", "TF")]
data_wide <- spread(df_libraryA, clusterID, mean.scaled.tp53)

data_wide_upper <- data_wide
data_wide_upper$seq <- toupper(data_wide$seq)

write.csv(data_wide[, c("State_3E", "seq", "CRS")], 
          "/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/upper_LibA_wide_pivot_state3.csv",
          row.names=FALSE)

# SIMPLE REGRESSION : Save state3E without upper case and IDS (model that sees only backgound)
df_libraryA <- libA[, c("clusterID", "seq", "mean.scaled.tp53","CRS")]
data_wide <- spread(df_libraryA, clusterID, mean.scaled.tp53)
write.csv(data_wide[, c("State_3E", "seq", "CRS")],
          "/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/no_upper_LibA_wide_pivot_state3.csv", row.names=FALSE)


# MULTIHEAD REGRESSION : Save all states with upper case and IDS (for model that sees the whole sequence)
df_libraryA <- libA[, c("clusterID", "seq", "mean.scaled.tp53","CRS")]
data_wide_upper <- spread(df_libraryA, clusterID, mean.scaled.tp53)
write.csv(data_wide_upper,
          "/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/upper_LibA_wide_pivot_all_states.csv", row.names=FALSE)

# MULTIHEAD REGRESSION : Save all states without upper case and IDS (for model that sees the only background)
df_libraryA <- libA[, c("clusterID", "seq", "mean.scaled.tp53","CRS")]
data_wide <- spread(df_libraryA, clusterID, mean.scaled.tp53)
write.csv(data_wide,
          "/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/no_upper_LibA_wide_pivot_all_states.csv", row.names=FALSE)

# MULTIHEAD REGRESSION : Save all states and randomized upper cases where the motifs are and IDS (for model that sees the background and randomized motifs)
df_libraryA <- libA[, c("clusterID", "seq", "mean.scaled.tp53","CRS")]
data_wide <- spread(df_libraryA, clusterID, mean.scaled.tp53)

# Function to replace upper case nucleotides with random lower case nucleotides
replace_uppercase_with_random <- function(sequence) {
  upper_nucleotides <- c("A", "T", "G", "C")
  lower_nucleotides <- c("a", "t", "g", "c")
  
  char_vector <- unlist(strsplit(sequence, ''))
  upper_indices <- which(char_vector %in% upper_nucleotides)
  
  if (length(upper_indices) > 0) {
    selected_lower <- sample(lower_nucleotides, length(upper_indices), replace = TRUE)
    char_vector[upper_indices] <- selected_lower
  }
  
  return(paste(char_vector, collapse = ''))
}
# Apply the function to the DNA column
data_wide$seq <- sapply(data_wide$seq, replace_uppercase_with_random)
write.csv(data_wide,
          "/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/randomized_motifs_no_upper_LibA_wide_pivot_all_states.csv", row.names=FALSE)


# MULTIHEAD REGRESSION : Save all states and motifs with 2bp of extension are and IDS (for model that sees only the motifs and 2bps of background on each side)
df_libraryA <- libA[, c("clusterID", "seq", "mean.scaled.tp53","CRS")]
data_wide <- spread(df_libraryA, clusterID, mean.scaled.tp53)
# Extend upper case in both directions of the sequence
# Assuming your dataframe is named 'df' and the column is named 'seq'
data_wide$seq <- sapply(strsplit(data_wide$seq, ""), function(x) {
  # Identify the positions of uppercase letters
  upper_pos <- which(toupper(x) == x)
  
  # Extend the uppercase segments by 2 positions
  for (pos in upper_pos) {
    if (pos - 2 >= 1) x[pos - 2] <- toupper(x[pos - 2])
    if (pos - 1 >= 1) x[pos - 1] <- toupper(x[pos - 1])
    x[pos + 1] <- toupper(x[pos + 1])
    x[pos + 2] <- toupper(x[pos + 2])
  }
  
  return(paste(x, collapse = ""))
})

# Now, df$seq contains the modified sequences

write.csv(data_wide,
          "/home/felix/cluster/fpacheco/Data/Robert_data/processed_data/2bp_upper_LibA_wide_pivot_all_states.csv", row.names=FALSE)

