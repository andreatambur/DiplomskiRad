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
library(pROC)
library(ROSE)

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

# kreiranje modela krosvalidacijom

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

# KREIRANJE TRECEG MODELA

# kreiranje modela koriscenjem balansiranog dataset-a
trainData$smoking <- factor(trainData$smoking, levels = c(0, 1), labels = c("No", "Yes"))
testData$smoking <- factor(testData$smoking, levels = c(0, 1),labels = c("No", "Yes"))

# kreiranje kontrolnog modela koji koristi metod unakrsne krosvalidacije s ponavljanjem radi vece pouzdanosti
ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down")
# downSample
set.seed(1010)
down_inside <- train(x = trainData[,-16], y = trainData$smoking, method = "rpart",
                     metric = "ROC", trControl = ctrl, tuneGrid = cpGrid)

# upSample
ctrl$sampling <- "up"
set.seed(1010)
up_inside <- train(x = trainData[,-16], y = trainData$smoking, method = "rpart",
                   metric = "ROC", trControl = ctrl, tuneGrid = cpGrid)

# ROSE
ctrl$sampling <- "rose"
set.seed(1010)
rose_inside <- train(x = trainData[,-16], y = trainData$smoking, method = "rpart",
                     metric = "ROC", trControl = ctrl, tyuneGrid = cpGrid)

# originalan model bez uzrokovanja
ctrl$sampling <- NULL
set.seed(1010)
orig_fit <- train(x = trainData[,-16], y = trainData$smoking, method = "rpart",
                  metric = "ROC", trControl = ctrl,tuneGrid = cpGrid)

# pravimo listu sa svim modelima i uporedjujemo ih
inside_models <- list(original = orig_fit,
                      down = down_inside,
                      up = up_inside,
                      ROSE = rose_inside)
inside_resampling <- resamples(inside_models)
summary(inside_resampling, metric = "ROC")
# ROC 
#               .Min.   1st Qu.     Median      Mean      3rd Qu.      Max.    NA's
# original 0.7745605  0.7861593* 0.7918902* 0.7915344* 0.7968416* 0.8077616*     0
# down     0.7719302  0.7825974  0.7883817  0.7881037  0.7926108  0.8090818*    0
# up       0.7589179  0.7826196  0.7877585  0.7868085  0.7927252  0.8026215     0
# ROSE     0.7763761* 0.7857924* 0.7909512* 0.7909360* 0.7964803* 0.8063076     0

# originalni model daje bolje rezultate od svih ostalih, ali za njim je sledeci ROSE

png("stablo_odlucivanja3.png", width = 1700, height = 1200, res = 150)
rpart.plot(rose_inside$finalModel, 
           extra = 106,   
           cex = 0.8,         
           main = "Stablo odlučivanja") 
dev.off()

# kreiranje predikcije, matrice konfuzije i evalucionih metrika
tree3.pred <- predict(rose_inside$finalModel, newdata = testData, type = "class")

tree3.cm <- table(true = testData$smoking, predicted = tree3.pred)
tree3.cm
#        predicted
# true    No  Yes
# No     3441 1492
# Yes    758 2105

eval.tree3 <- getEvaluationMetrics(tree3.cm)
eval.tree3
# Accuracy Precision    Recall        F1 
# 0.7113905 0.5852099 0.7352428 0.6517028  


# uporedjivanje rezultata modela i tumacenje
data.frame(rbind(eval.tree1, eval.tree2,eval.tree3), row.names = c("Prvi model","Drugi model","Treci model"))

#             Accuracy  Precision    Recall        F1
# Prvi model  0.7142124 0.6241689 0.5574572 0.5889299
# Drugi model 0.7243458 0.6107320 0.6877401 0.6469525
# Treci model 0.7113905 0.5852099 0.7352428 0.6517028

# accuracy je procenat tacnih predikcija ukupnog broja observacija
# mozemo videti da je rezultat za prvu metriku kod sva tri modela priblizno isti

# precision je procenat observacija koje su pozitivne od onih ukupnog broja koje smo predvideli da su pozitivne
# za prvi model vidimo da je procenat 62,41%, dok je kod drugog modela 61,07%, a treci model je nesto nizi 58,52%

# recall je procenat observacija koje smo predivdeli da su pozitivne u odnosu na ukupan broj stvarno pozitivnih
# istakao se treci izbalansirani model sa najvecim procentom od 73,52%

# F1 sluzi za evaluaciju modela kada su precision i recall u balansu, govori koliko je dobar model
# mozemo reci da je iznad 70% prihvatljiv model, te i zakljuciti da nijedan od nasih modela nije pozeljno upotrebljiv


# ZNACAJNOST ATRIBUTA
# pokazuje koliko je svaka promenljiva doprinela u procesu donošenja odluka unutar modela,
# obično meri na osnovu toga koliko su one doprinele smanjenju greške ili povećanju tačnosti modela

rose_inside$finalModel$variable.importance

# hemoglobin       height.cm.              Gtp       weight.kg. serum.creatinine        waist.cm. 
# 2380.212461      1622.487593       986.758253       402.667822       298.102809       178.680120 
# triglyceride              HDL    dental.caries       relaxation              age         systolic 
# 163.101734        42.537413        39.458227        26.671174        24.582964         5.396858 

# iz prilozenih rezultata mozemo videti da najvecu znacajnost ima varijabla hemoglobin
# ne zaostaju za njom i visina i Gtp, te slede ostale, a najmanju znacajnost ima gornji pritisak






