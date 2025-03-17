#functions for sequence design
load(url("https://figshare.com/ndownloader/files/53009321"))
load("RF_results.rda")

#this might create problems because of different universe of packages
source("Generate_Sequences.R")
source("../query_models.R")


loss <- function(current,target) {
  if (sd(target) == 0) {
    return(-sum((target-current)^2))
  } else {
    return( cor(target,current) - sum((target-current)^2) )
  }
  
}

#to get a starting sequence from design
#based on RF/001_basic_RF.R
#useMeasurements: if TRUE, use measured data values and imputation from RF where data is missing. If FALSE, use test set predictions of RF
#nDesigns: Extract the top n design from the space 
#nDesignReps: For each design, deliver n sequences. If set to 0, the function will output the actual sequence used in the data. If no sequence is available, it will issue a warning and create one sequence
get_starting_sequence <- function(target, useMeasurements = T, nDesigns = 6, nDesignReps = 0, returnDesignDF = F, returnFullDF = F) {
  if(useMeasurements) {
    refDF <- imputed.wide
  } else {
    refDF <- predicted.wide
  }
  
  usecols <- c( "State_1M","State_2D","State_3E","State_4M","State_5M","State_6N","State_7M" )
  refDF$score_rf <- apply(refDF[,usecols],1,loss, target= target)
  if (returnFullDF) return(refDF)
  
  refDF <- refDF[order(refDF$score_rf, decreasing = T),]
  use <- head(refDF, n = nDesigns)
  if (returnDesignDF) return(use)
  print(use)
  #need to change here if the structure changes with LibB (include sequences that have never been measured)
  if (nDesignReps == 0) {
      result <- sapply(use$CRS, function(x) ifelse(grepl("LibH",x), 
                                                    with( mpra.data$HSPC.libA$DATA, Seq[CRS==x]),
                                                    with( mpra.data$HSPC.libB$DATA, substr(Seq[CRS==x],17,262))))
      use$spacer <- sprintf("%d", use$spacer)
      names(result) <- apply(use[,c(1,3:11)], 1, paste, collapse="_")
      return(result)
  } else {
    #run Robert's design script
    #change the format I used to the one used by Robert
    for (i in 1:nrow(use)) {
      if (use$TF1.name[i] == use$TF2.name[i]) {
        use$TF2.affinity[i] <- NA
        use$TF2.orientation[i] <- NA
        use$TF2.name[i] <- NA
      } else {
        use$TFnumber[i] <- use$TFnumber[i]/2
      }
    }
    
    designs <- convert_dataframe(use)
    
    general_params <<- list(
      draws = 100,
      fill_string = TRUE, 
      seq_number = nDesignReps,
      filter_length = T, 
      seq_length=246
    )
    
    designs_checked <- check_sampling_library(designs)
    
    
    result <- unlist(getLibrary(designs_checked))
    return(result)
    
  }
  
}


get_all_mutations <- function(seq) {
  seq <- toupper(seq)
  out <- lapply(1:nchar(seq), function(i) {
    sapply(c("A","C","G","T"), function(ch) {
      cur = substr(seq, i, i)
      if (cur == ch) return(NULL) else return(sprintf("%s%s%s",substr(seq, 1, i-1), ch,substr(seq,i+1,nchar(seq)) ))
    })
  })
  unlist(out)
}

evolve_sequence <- function(target, steps, init = NULL, ninit = 10,adapter1 = "", adapter2 = "",nchar=246, eval.by.deep =TRUE, eval.by.mech = TRUE, explain = TRUE, deep.ensembl.size = 1, port.mech= 4567, port.deep = 4568, deep.server = "10.44.1.92") {
  if (is.null(init)) {
    init <- replicate(ninit, paste(sample(c("A","C","G","T"),nchar-nchar(adapter1)-nchar(adapter2),TRUE),collapse=""))
    init <- paste0(adapter1,init,adapter2)
  }
  out <- c()
  for (i in 1:length(init)) {
    s <- init[i]
    for (j in 1:steps) {
      cat("Init ", i, " step ", j,"\n")
      candidates <- paste0(adapter1, get_all_mutations( substr(s, 1+nchar(adapter1), nchar(s)-nchar(adapter2) )), adapter2)
      
      if (eval.by.mech) {
        cat("Evaluating mech model\n")
        pred_mech <- predict_mech_model(candidates, port=port.mech)
        score_mech <- apply(pred_mech,1,loss, target= target)
      }
      
      if (eval.by.deep) {
        cat("Evaluating deep model\n")
          pred_deep <- predict_deep_model(candidates, ensembl_size = deep.ensembl.size, port = port.deep, server = deep.server)
        score_deep <- apply(pred_deep,1,loss, target= target)
      }
      
      if (eval.by.deep) score_combined <- score_deep
      if (eval.by.mech) score_combined <- score_mech
      if (eval.by.deep & eval.by.mech) score_combined <- score_deep + score_mech
      
      s <- candidates[which.max(score_combined)]
      
      if (eval.by.deep & eval.by.mech) {
        cat("Deep score: ", score_deep[which.max(score_combined)], " mech score: ", score_mech[which.max(score_combined)],"\n")
      } else if (eval.by.deep) {
        cat("Deep score: ", score_deep[which.max(score_combined)],"\n")
      } else if (eval.by.mech) {
        cat("Mech score: ", score_mech[which.max(score_combined)],"\n")
      }
      
    }
    out <- c(out, s)
    if (eval.by.deep) print(pred_deep[which.max(score_combined),])
    if (eval.by.mech) print(pred_mech[which.max(score_combined),])
    
  }
  return(out)
}




evolve_mcmc <- function(target, steps, init = NULL, ninit = 10,nchar=246, adapter1 = "", adapter2 = "", deep.ensembl.size = 1, port.mech= 4567, port.deep = 4568, deep.server = "10.44.1.92") {
  if (is.null(init)) {
    init <- replicate(ninit, paste(sample(c("A","C","G","T"),nchar-nchar(adapter1)-nchar(adapter2),TRUE),collapse=""))
    init <- paste0(adapter1,init,adapter2)
  }
  out <- c()
  bases <- c("A","C","G","T")
  
  for (i in 1:length(init)) {
    s <- init[i]
    pred_deep <- predict_deep_model(s, ensembl_size = deep.ensembl.size, port = port.deep, server = deep.server)
    score_last <- apply(pred_deep,1,loss, target= target)
    score_initial <- score_last
    accepted <- 0
    rejected <- 0
    for (j in 1:steps) {
      position <- sample((nchar(adapter1)+1):(nchar(s)-nchar(adapter2)), 1)
      mutation <- sample(bases[bases!=substr(s,position,position)],1)
      if (position == 1) {
        candidate <- paste0(mutation, substr(s, 2, nchar(s)))
      } else if (position == nchar(s)) {
        candidate <- paste0(substr(s,1,nchar(s)-1),mutation)
      } else {
        candidate <- paste0(substr(s, 1, position - 1), mutation, substr(s,position+1, nchar(s)))
      }
      
     

        pred_deep <- predict_deep_model(candidate, ensembl_size = deep.ensembl.size, port = port.deep, server = deep.server)
        score_deep <- apply(pred_deep,1,loss, target= target)

      if (score_deep > score_last | runif(1) < 0.1) {
        accepted <- accepted + 1
        s <- candidate
        score_last <- score_deep
      } else {
        rejected <- rejected + 1
      }
      #if (j %% 100 == 0) cat(sprintf("Task %s, init %d, step %d, acceptance rate %.2f %%, score: %.3f -> %.3f\n", paste(target,collapse = ","), i, j, 100*accepted/j, score_initial, score_last ))
    }
    out <- c(out, s)
    
  }
  return(out)
}
