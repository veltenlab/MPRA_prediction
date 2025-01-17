

predict_deep_model_batch <- function(sequences, batch, server = "127.0.0.1", port=4568, ensembl_size = 10) {
  if (!all(nchar(sequences) == 246)) stop("All sequences need to be length 246 characters")
  if (!all(batch %in% c("LibA","LibH","LibB","LibV"))) stop("Batch needs to be one of LibA,LibH,LibB,LibV")
  if (ensembl_size > 10) stop("Max ensembl_size is 10")
  
  con_deep <- socketConnection(server, port = port, open = "r+b", blocking = TRUE)
  writeChar(sprintf("%02d",ensembl_size), con_deep,eos=NULL)
  pred_deep <- sapply(sequences, function(ss) {
    writeChar(paste0( toupper(ss), ",", batch, whitespace(280-nchar(batch)-246-1)), con_deep,eos=NULL)
    y <- readLines(con_deep, n =ensembl_size)
    if (ensembl_size==1) return(as.numeric(strsplit(y, ",")[[1]])) else {
      out <- sapply(strsplit(y, ","), as.numeric)
      return(apply(out,1,mean)  )
    }
    
  })
  pred_deep <- t(pred_deep)
  close(con_deep)
  colnames(pred_deep) <- c("State_1M", "State_2D", "State_3E", "State_4M", "State_5M", "State_6N", "State_7M")
  return(pred_deep)
}