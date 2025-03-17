
#Helper functions
'%nin%' = Negate('%in%')
convert_to_universal <- function(tfbs_list) {
  if (tfbs_list@strand == "*") { tfbs_list@strand = "+-"
  } else {
    tfbs_list@strand = tfbs_list@strand
  }
  out <- convert_motifs(tfbs_list)
  return(out)
}
rev_comp <- function(nucSeq) {
  out <- stri_reverse(chartr("acgtACGT", "tgcaTGCA", nucSeq))
  names(out) <- names(nucSeq)
  return(out)
}

#Load Input
mouse_motifs_selected <- readRDS("TFBS_files/TFBS_database/221020_mouse_selected_44TFBS_medoid_motifs.rds")
gene_names <- readRDS("TFBS_files/TFBS_database/221020_mouse_selected_44TFBS_gene_names.rds")
motifs_shuffled <- readRDS("TFBS_files/TFBS_database/221020_shuffled_motifs.rds")
spacer_seq_list <- readRDS("TFBS_files/Spacer/221219_merged_spacer_list_LongSpacers.rds")

#ConvertToUniversal
mouse_motifs_selected_universal <- lapply(mouse_motifs_selected, function(x) {
  motif <- convert_to_universal(x)
  motif@type <- "PPM"
  motif
})

#ConvertDataframe
convert_dataframe <- function(df) { 
  #Select TF columns
  tf_columns <- grep("^TF\\d+\\.name$", names(df), value = TRUE)
  tf_columns <- gsub(pattern = "name",replacement = "", tf_columns)
  #Make conversion and safe parameters
  result_list <- lapply(seq_len(nrow(df)), function(i) {
    lapply(tf_columns, function(tf) {
      tf_entry <- list(tf_factor = as.character(df[[paste0(tf, "name")]][i]),
                       number_of_sites = as.integer(df$TFnumber[i]),
                       affinity_range = as.numeric(df[[paste0(tf, "affinity")]][i]),
                       orientation = as.character(df[[paste0(tf, "orientation")]][i]),
                       spacing = ifelse(paste0(tf, "spacer") %in% colnames(df),as.character(df[[paste0(tf, "spacer")]][i]), as.character(df$spacer[i])  ),
                       TForder = as.character(df$TForder[i])
      )
      if (!any(sapply(tf_entry, is.na))) { return(tf_entry) } else { return(NULL) }
    })
  })
  # Filter out NULL entries (entries with NA)
  result_list <- lapply(result_list, function(entry) entry[!sapply(entry, is.null)])
  result_list
}

#GenerateMotifAffinity
motif_generate_affinity <- function(tf, affinity_range ) {
  out <- motifs_shuffled[[tf]]
  as.character(out$motif[ceiling(runif(1,1-affinity_range-0.025, 1-affinity_range+0.025) * nrow(out))])
}

#TFBSSpacerList
#Get TFBS
get_possible_TFBSsequence <- function(tf_factor, affinity_range, orientation) {
  tfbs_motif <- mouse_motifs_selected[[tf_factor]]
  motif_seq <- c()
  if (orientation == "random") { orientation <- sample(x = c("fwd", "rev"), size = 1)}
  if (affinity_range == 1) {
    motif_seq <- paste0(apply(tfbs_motif@profileMatrix, 2, function(x) names(which.max(x))), collapse = "")
    names(motif_seq) <- paste0("tf_", paste0(tf_factor, collapse = ""), "_aff_NULL_ori_", paste0(unlist(orientation), collapse = ""))  
  } else {
    motif_seq <- motif_generate_affinity(tf_factor, affinity_range)
    names(motif_seq) <- paste0("tf_", paste0(tf_factor, collapse = ""), "_aff_c(", paste0((affinity_range), sep= "" ,collapse = ","), ")_ori_", paste0(unlist(orientation), collapse = ""))
  }
  if (orientation == "rev") { motif_seq <- rev_comp(motif_seq) }
  return(motif_seq)
}

#Get spacer
get_spacer_seq <- function(spacing) {
  spacer_seq <- tolower(sample(x = spacer_seq_list[[spacing]], size = 1)) 
  names(spacer_seq) <- paste0("spa_", spacing) 
  return(spacer_seq)
}

#Make list for TFBS & spacer
get_TFBS_spacer <- function(tf_factor, affinity_range, orientation, spacing, number_of_sites) {
  TFBS_spacer_list <- list()
  TFBS_spacer_list <- lapply(1:number_of_sites, function(x){ 
    TFBS_spacer_list[[x]] <- list("TFBS" = get_possible_TFBSsequence(tf_factor, affinity_range, orientation), "spacer" = get_spacer_seq(spacing))
  })
  return(TFBS_spacer_list)
}

#OrderTfsFunctions
consecutive_min <- function(x) {
  out <- list()
  for(i in 1:length(x)) {
    min_i <- min(x)
    out[[i]] <- min_i
    x <- x[!x %in% min_i]
  }
  return(unlist(out))
}

order_TFs_alternate <- function(x, number_tfs) {
  x <- split(x, ceiling(seq_along(x)/(length(x)/number_tfs)))
  out <- apply(as.matrix(sapply(x, function(y) { out <- consecutive_min(y) }, simplify = T)), 1, FUN = function(z) z) %>% as.vector()
  return(out)
} 

order_Block_name <- function(x, number_tfs) {
  x <- split(x, ceiling(seq_along(x)/(length(x)/number_tfs)))
  out <- apply(as.matrix(sapply(x, function(y) { out <- c(min(y), min(y)+1) }, simplify = T)), 2, FUN = function(z) z) %>% as.vector()
  return(out)
}

#GetLibrary
getLibrary <- function(x) { sapply(x, function(x) {
  #Get TFBS & spacer combinations
  tf_seq <-  replicate(n = general_params$draws, expr = unlist(lapply(x, function(y) { get_TFBS_spacer(tf_factor = y$tf_factor, affinity_range = y$affinity_range, orientation = y$orientation, spacing = y$spacing, number_of_sites = y$number_of_sites)})))
  tf_seq <- apply(tf_seq, 2, as.list)
  
  #Save number of TF repeats
  number_elements <- as.numeric(unique(replicate(n = 1, expr = unlist(lapply(x, function(y) { y$number_of_sites})))))
  TForder <- as.character(unique(replicate(n = 1, expr = unlist(lapply(x, function(y) { y$TForder})))))
  
  #Generate GRE sequence
  possible_sequences <- lapply(tf_seq, function(x) {paste_gre(x = x, number_tfs = number_elements, orderTF = TForder)})
  
  #Add filling seq to have all sequences same length
  if (general_params$fill_string == TRUE) {
    filling_seq <- replicate(n = general_params$draws, expr = paste0(sample(c("a", "c", "g", "t"), size = (general_params$seq_length-unique(nchar(possible_sequences))-30), replace = T), collapse=""))
    #The merging consists out of backbone1_seq(Oligo_left), the filling seq, the GRE seq and the second part of the backbone_seq
    backbone_seq1 <- "aggaccggatcaact"
    backbone_seq2 <- "cattgcgtgaaccga"
    possible_sequences <- setNames(paste0(backbone_seq1, filling_seq, possible_sequences, backbone_seq2), lapply(possible_sequences, function(x) {names(x)} ))
  } else {
    possible_sequences <- sapply(possible_sequences , function(x) {x}, simplify = T, USE.NAMES = T)  
  } # Modify names into fasta format  
  names(possible_sequences) <- paste0(">",seq(1:length(possible_sequences)), "::",names(possible_sequences))
  #Safe TFs for filtering them out in Fimo
  TF_comb_names <- unique(str_replace(string = str_replace(string = as.character(str_extract_all(string = unique(names(possible_sequences)), pattern = "tf_(.*?)_aff", simplify = T)), pattern = "tf_", replacement = ""), pattern = "_aff", replacement = ""))
  #Convert to Biostring
  possible_sequences_biostring <- BStringSet(x=possible_sequences, use.names=TRUE)
  #runFimo
  fimo_res <- runFimo(sequences = possible_sequences_biostring, motifs = mouse_motifs_selected_universal[names(mouse_motifs_selected_universal) %nin% TF_comb_names],  bfile = "uniform", max_stored_scores = 1000000)
  #Filter sequences with Fimo data
  if(is.null(fimo_res) == TRUE) {
    final_seq <- sample(x = possible_sequences, size = general_params$seq_number, replace = TRUE)
    names(final_seq) <- gsub(".*::","",names(final_seq))
  } else {
    if (length(unique(fimo_res@seqnames)) == general_params$draws) { 
      use <- which(gsub("::.*", "",names(possible_sequences)) %in% names(which(table(fimo_res@seqnames@values) == min(table(fimo_res@seqnames@values)))))
      final_seq <- sample(x = possible_sequences[use], size = general_params$seq_number, replace = TRUE)
      names(final_seq) <- gsub(".*::","QCFailure___",names(final_seq)) 
    } else {
      use <- which(gsub("::.*", "",names(possible_sequences)) %nin% fimo_res@seqnames@values)
      final_seq <- sample(x = possible_sequences[use], size = general_params$seq_number, replace = TRUE)
      names(final_seq) <- gsub(".*::","",names(final_seq))  
    }}
  names(final_seq) <- paste0(names(final_seq), "__TForder_", TForder)
  #Return sequence
  return(list(final_seq))}
  , simplify = T)
}

paste_gre <- function(x, number_tfs, orderTF ) {
  len <- length(x)
  out <- c()
  name_out <- c()
  seqIndex <- seq(1, len, by=2)
  if(orderTF %in% "Alternate") { 
    seqIndex <- order_TFs_alternate(seqIndex, length(seqIndex)/number_tfs)
  }
  if(orderTF %in% "Block" | orderTF %in% "Alternate" ) {
    for(i in seqIndex) {
      out <-  paste0(append(out, paste0(x[c(i,i+1)], collapse = "")), collapse = "")
      name_out <- append(name_out, names(x)[c(i,i+1)])
    }
    ifelse(orderTF %in% "Alternate", name_out <- name_out[1:(length(name_out)/number_tfs)], name_out <- name_out[order_Block_name(1:length(name_out),(length(name_out)/number_tfs/2))]) 
  } else {
    print("Select proper TFBS order - 'Block' or 'Alternate'")
  }
  out <- str_remove(string = out, pattern = '[:lower:]+$')
  names(out) <- paste0(paste0(name_out, collapse = "_._"), "__TFnumber_",  (length(seqIndex)/number_tfs), "__TFrepeats_", number_tfs ) 
  return(out)
}

#Check if a design like this with the specified length can be designed
check_sampling_library <- function(out) {
  out_temp <- sapply(out, function(x) {
    tf_seq <-  replicate(n = 1, expr = unlist(lapply(x, function(y) { get_TFBS_spacer(tf_factor = y$tf_factor, affinity_range = y$affinity_range, orientation = y$orientation, spacing = y$spacing, number_of_sites = y$number_of_sites)}))) 
    tf_seq <- apply(tf_seq, 2, as.list) 
    number_elements <- as.numeric(unique(replicate(n = 1, expr = unlist(lapply(x, function(y) { y$number_of_sites})))))
    TForder <- as.character(unique(replicate(n = 1, expr = unlist(lapply(x, function(y) { y$TForder})))))
    possible_sequences <- lapply(tf_seq, function(x) {paste_gre(x = x, number_tfs = number_elements, orderTF = TForder)})
  })
  out_temp <- (nchar(out_temp) <= general_params$seq_length-30) # 30 bp for adaptors sequences
  out <- out[out_temp]
  return(out)
}

