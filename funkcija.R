getEvaluationMetrics <- function(cm) {
  TP <- cm[2,2] # tacno predvidjeni pozitivni
  TN <- cm[1,1] # tavno predvidjeni negativni
  FP <- cm[1,2] # netacno predvidjeni pozitivni
  FN <- cm[2,1] # netacno predvidjeni negativni
  
  accuracy = sum(diag(cm)) / sum(cm) # procenat tacnih predikcija
  precision <- TP / (TP + FP) # procenat tacno predvidjenih pozitivnih od ukupno svih predvidjenih pozitivnih
  recall <- TP / (TP + FN) # procenat tavno predvidjenih pozitivnih od ukupno svih pozitivnih
  F1 <- (2 * precision * recall) / (precision + recall) # evaluacija modela
  
  c(Accuracy = accuracy, 
    Precision = precision, 
    Recall = recall, 
    F1 = F1)
  
}