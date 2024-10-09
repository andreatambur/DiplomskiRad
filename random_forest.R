# ucitavanje sredjenog dataset-a
preprocessed_dataset<- readRDS("preprocessed_dataset.RDS")
data <- preprocessed_dataset

# ucitavanje funkcije
source("funkcija.R")

# instaliranje paketa i ucitavanje biblioteka neophodnih za rad
# install.packages('caret')
install.packages('randomForest')
library(randomForest)
library(caret)
library(pROC)
library(rpart)
library(ROSE)

str(data)

# kreiranje trening i test setova
set.seed(1010) 
i <- createDataPartition(data$smoking, p = 0.8, list = FALSE)
trainData<- data[i, ]
testData <- data[-i, ]

# KREIRANJE PRVOG MODELA

# kreiranje modela, predikcije, matrice konfuzije i evalucione metrike
rf1 <- randomForest(smoking ~ ., data = trainData)
rf1.pred <- predict(object = rf1, newdata = testData, type = "class")
rf1.cm <- table(true = testData$smoking, predicted = rf1.pred)
rf1.cm
#        predicted
# true    0    1
# 0     4092  841
# 1      806 2057
eval.rf1<-getEvaluationMetrics(rf1.cm)
eval.rf1
# Accuracy Precision    Recall        F1 
# 0.7887378 0.7097999 0.7184771 0.7141121 

# KREIRANJE DRUGOG MODELA

# kreiranje modela koriscenjem balansiranog dataset-a
trainData$smoking <- factor(trainData$smoking, levels = c(0, 1), labels = c("No", "Yes"))
testData$smoking <- factor(testData$smoking, levels = c(0, 1),labels = c("No", "Yes"))

# kreiranje kontrolnog modela koji koristi metod krosvalidacije s ponavljanjem radi vece pouzdanosti
ctrl <- trainControl(method = "cv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down")
# postupak krosvalidacije u kombinaciji sa random forest algoritmom i obimnim data setom, zahteva mnogo vremena za izvrsenje
# te smo postupak realizovali nad podskupom podataka
trainDataSubSet<-trainData[sample(nrow(trainData), 1000),]

# downSample
set.seed(1010)
down_inside <- train(x = trainDataSubSet[,-16], y = trainDataSubSet$smoking, method = "rf",
                     metric = "ROC", trControl = ctrl)

# upSample
ctrl$sampling <- "up"
set.seed(1010)
up_inside <- train(x = trainDataSubSet[,-16], y = trainDataSubSet$smoking, method = "rf",
                   metric = "ROC", trControl = ctrl)

# ROSE
ctrl$sampling <- "rose"
set.seed(1010)
rose_inside <- train(x = trainDataSubSet[,-16], y = trainDataSubSet$smoking, method = "rf",
                     metric = "ROC", trControl = ctrl)

# originalan model bez uzrokovanja
ctrl$sampling <- NULL
set.seed(1010)
orig_fit <- train(x = trainDataSubSet[,-16], y = trainDataSubSet$smoking, method = "rf",
                   metric = "ROC", trControl = ctrl)

# pravimo listu sa svim modelima i uporedjujemo ih
inside_models <- list(original = orig_fit,
                      down = down_inside,
                      up = up_inside,
                      ROSE = rose_inside)
inside_resampling <- resamples(inside_models)
summary(inside_resampling, metric = "ROC")
# ROC 
#                Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# original 0.7082797 0.8016098 0.8152479 0.8184686 0.8466016 0.9006864    0
# down     0.7295152 0.7880738 0.8171386 0.8151070 0.8403540 0.9017589    0
# up       0.7162162 0.7950450 0.8163257 0.8163674 0.8489218 0.9045474    0
# ROSE     0.7012012 0.7739337 0.8068934 0.7999499 0.8286290 0.9062634    0

# najbolji je upSample

# kreiranje predikcije, matrice konfuzije i evalucionih metrika
rf2.pred <- predict(up_inside$finalModel, newdata = testData, type = "class")

rf2.cm <- table(true = testData$smoking, predicted = rf2.pred)
rf2.cm
#        predicted
# true    No  Yes
# No    3674 1259
# Yes    897 1966

eval.rf2 <- getEvaluationMetrics(rf2.cm)
eval.rf2
# Accuracy Precision    Recall        F1 
# 0.7234479 0.6096124 0.6866923 0.6458607 

# KREIRANJE TRECEG MODELA

m<- sqrt(ncol(data))
grid <- expand.grid(mtry = c(m-2,m-1,m,m+1,m+2))

train_control <- trainControl(method = "cv",
                              number = 5,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

model_rf <- train(x = trainDataSubSet[,-16], y = trainDataSubSet$smoking, method = "rf", metric = "ROC",
                  tuneGrid = grid, trControl = train_control)
best_mtry <- model_rf$bestTune$mtry
# best_mtry je 3
model_rf$finalModel
# Confusion matrix:
#      No Yes class.error
# No  525 103   0.1640127
# Yes 141 231   0.3790323

# kreiranje predikcije, matrice konfuzije i evalucionih metrika
rf3 <- randomForest(smoking ~., data = trainDataSubSet, mtry = best_mtry) 
rf3.pred <- predict(object = rf3, newdata = testData, type = "class")

rf3.cm <- table(true = testData$smoking, predicted = rf3.pred)
rf3.cm
#        predicted
# true    No  Yes
# No    3905 1028
# Yes   1154 1709
eval.rf3 <- getEvaluationMetrics(rf3.cm)
eval.rf3
# Accuracy Precision    Recall        F1 
# 0.7201129 0.6244063 0.5969263 0.6103571 


# uporedjivanje rezultata modela i tumacenje
data.frame(rbind(eval.rf1,eval.rf2,eval.rf3), row.names = c("Prvi model","Drugi model","Treci model"))

#               Accuracy Precision    Recall        F1
# Prvi model  0.7887378 0.7097999 0.7184771 0.7141121
# Drugi model 0.7234479 0.6096124 0.6866923 0.6458607
# Treci model 0.7201129 0.6244063 0.5969263 0.6103571

# prvi model nad nebalansiranim podacima daje znatno bolje rezulate nego drugi i treci, u pogledu svake metrike














