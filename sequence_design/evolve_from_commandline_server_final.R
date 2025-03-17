
#Packages
suppressPackageStartupMessages({
  library(tibble)
  library(dplyr)
  library(Matrix)
  library(TFBSTools)
  library(Biostrings)
  library(S4Vectors)
  library(universalmotif)
  library(seqinr)
  library(stringr)
  library(stringi)
  library(memes)
  library(purrr)
  library(ggplot2)
  library(patchwork)
  library(tidyr)
})


source("design_sequence.R")

args <- commandArgs(trailingOnly = T)
task <- as.numeric(strsplit(args[1],",")[[1]])
nsteps <- as.integer(args[2])
nsteps.mcmc <- as.integer(args[3])
port <- as.integer(args[4])
outpath <- args[5]

run_evolutuion <- function(task, nsteps = 10,nsteps.mcmc = 10000, port = 4567) {
  cat("======================================\n")
  cat("Now working on ", paste(task, collapse = ","),"\n")
  cat("Using port, ", port,"\n")
  cat("======================================\n")
  #sometimes throws out the same sequence for all 
    
  #get starting sequence from random forest predictions
    start_predicted <- get_starting_sequence(task, nDesigns = 5, nDesignReps = 3,useMeasurements = F)
    
    backbone_seq1 <- "aggaccggatcaact"
    backbone_seq2 <- "cattgcgtgaaccga"
    #global search
    evolved_predicted <- evolve_mcmc(task, steps = nsteps.mcmc, init = toupper(start_predicted), adapter1 =backbone_seq1, adapter2=backbone_seq2,port.deep = port, deep.ensembl.size = 3, deep.server = "127.0.0.1")
    evolved_random <- evolve_mcmc(task, ninit=15, steps = nsteps.mcmc, adapter1 =backbone_seq1, adapter2=backbone_seq2, port.deep=port, deep.ensembl.size = 3, deep.server = "127.0.0.1")

    #local search    
    evolved_predicted2 <- evolve_sequence(task, steps = nsteps, init = toupper(start_predicted),eval.by.mech=F, adapter1 =backbone_seq1, adapter2=backbone_seq2,port.deep = port, deep.ensembl.size = 3, deep.server = "127.0.0.1")
    cat("6\n")
    evolved_random2 <- evolve_sequence(task, ninit=15, steps = nsteps, adapter1 =backbone_seq1, eval.by.mech=F,adapter2=backbone_seq2,port.deep=port, deep.ensembl.size = 3, deep.server = "127.0.0.1")
    cat("7\n")

  tryCatch({
    output <- data.frame(strategy = c(rep("imputed",length(evolved_imputed)), rep("predicted",length(evolved_predicted)), rep("random",length(evolved_random)),
                                      rep("imputed",length(evolved_imputed2)), rep("predicted",length(evolved_predicted2)), rep("random",length(evolved_random2))),
                         search = c(rep("mcmc",length(evolved_imputed)), rep("mcmc",length(evolved_predicted)), rep("mcmc",length(evolved_random)),
                                    rep("optim",length(evolved_imputed2)), rep("optim",length(evolved_predicted2)), rep("optim",length(evolved_random2))),
                         design = c(names(start_imputed), names(start_predicted), rep(NA, length(evolved_random)),
                                    names(start_imputed2), names(start_predicted2), rep(NA, length(evolved_random2))),
                         model = "deep",
                         task = paste(task, collapse = ","),
                         seq = c(evolved_imputed, evolved_predicted,evolved_random, evolved_imputed2, evolved_predicted2,evolved_random2))
    predicted <- predict_deep_model(output$seq,ensembl_size = 10, port = port, server = "127.0.0.1")
    output$score  <- apply(predicted, 1, loss, target = task)
    output <- cbind(output, predicted)
  }, error = function(e) {
    save(evolved_imputed, evolved_predicted,evolved_random, evolved_imputed2, evolved_predicted2,evolved_random2, file = sprintf("%s/%s.rescued.rda" , outpath, paste(task, collapse="_")))
    stop(e)
  })
  
  return(output)
  
}

out <- run_evolutuion(task, nsteps, nsteps.mcmc, port)

saveRDS(out, file = sprintf("%s/%s.rds" , outpath, paste(task, collapse="_")))
