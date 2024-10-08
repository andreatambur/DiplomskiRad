# ucitavanje sredjenog dataset-a
preprocessed_dataset<- readRDS("preprocessed_dataset.RDS")
data <- preprocessed_dataset

# ucitavanje funkcije
source("funkcija.R")

# instaliranje paketa i ucitavanje biblioteka
install.packages('caret')
install.packages("rpart.plot")
library(rpart.plot)
library(caret)
library(rpart)

# pre svega potrebno je napraviti trening i test setove
set.seed(1010) 
i <- createDataPartition(data$smoking, p = 0.8, list = FALSE)
trainData <- data[i, ]
testData <- data[-i, ]

# KREIRANJE PRVOG MODELA
tree1 <- rpart(smoking ~ ., data = trainData, method = "class")

# crtanje i cuvanje grafikona stabla odlucivanja
png("stablo_odlucivanja1.png", width = 1200, height = 800, res = 150)
rpart.plot(tree1, 
           extra = 106,   
           cex = 0.8,         
           main = "Stablo odlučivanja") 
dev.off()

# kreiranje predikcije i matrice konfuzije, razmatranje evalucionih metrika
tree1.prediction<-predict(tree1, newdata=testData, type="class")
tree1.cm<-table(true=testData$smoking,predicted=tree1.prediction)
tree1.cm
# na glavnoj dijagonali matrice konfuzije nam se nalazi broj tacnih
# predikcija, a na sporednoj broj pogresnih predikcija 
#         predicted
# true    0    1
#  0    3972  961
#  1    1267 1596
eval.tree1<-getEvaluationMetrics(tree1.cm)
eval.tree1
# Accuracy Precision    Recall        F1 
# 0.7142124 0.6241689 0.5574572 0.5889299 


# KREIRANJE DRUGOG MODELA

# krosvalidacija i odredjivanje optimalne vrednosti cp
numFolds <- trainControl(method = "cv", number = 10)
cpGrid <- expand.grid(.cp=seq(from = 0.001, to = 0.05, by = 0.001))
set.seed(1010)
crossvalidation <- train(x = trainData[,-16], y = trainData$smoking, method = "rpart",
                         trControl = numFolds,tuneGrid = cpGrid)
crossvalidation
cpValue <- crossvalidation$bestTune$cp
# cp za svoju vrednost uzima 0.001


# kreiranje stabla 
tree2 <- rpart(smoking ~ ., data = trainData, method = "class", control = rpart.control(cp = cpValue))

png("stablo_odlucivanja2.png", width = 1200, height = 800, res = 150)
rpart.plot(tree2, extra = 106, cex = 0.8, main = "Stablo odlučivanja") 
dev.off()

# kreiranje predikcije i matrice konfuzije, tumacenje evalucionih metrika
tree2.pred <- predict(tree2, newdata = testData, type = "class")
tree2.cm <- table(true = testData$smoking,predicted = tree2.pred)
tree2.cm
#        predicted
# true    0    1
# 0     3678 1255
# 1      894 1969
eval.tree2 <- getEvaluationMetrics(tree2.cm)
eval.tree2
# Accuracy Precision    Recall        F1 
# 0.7243458 0.6107320 0.6877401 0.6469525 

# uporedjivanje rezultata modela i tumacenje
data.frame(rbind(eval.tree1, eval.tree2), row.names = c("Prvi model","Drugi model"))

# accuracy je procenat tacnih predikcija ukupnog broja observacija
# nas test set ukupno obuhvata 7796 obs. od kojih smo mi tacno predvideli kod prvog modela 71,4%, a kod drugog 72,4%

# precision je procenat observacija koje su pozitivne od onih ukupnog broja koje smo predvideli da su pozitivne
# za prvi model vidimo da je procenat 62,41%, dok je kod drugog modela 61,07%

# recall je procenat observacija koje smo predivdeli da su pozitivne u odnosu na ukupan broj stvarno pozitivnih
# prvi model 55,74%, dok drugi ima primetno bolje rezultate 68,77%

# F1 sluzi za evaluaciju modela kada su precision i recall u balansu, govori koliko je dobar model
# mozemo reci da je iznad 70% prihvatljiv model, te i zakljuciti da nijedan od nasih modela nije pozeljno upotrebljiv









