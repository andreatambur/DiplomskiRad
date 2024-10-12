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
# postupak unakrsne krosvalidacije u kombinaciji sa random forest algoritmom i obimnim data setom, zahteva mnogo vremena za izvrsenje
# te smo postupak realizovali obicnom krosvalidacijom i ogranicavanjem broja stabala na 100

# downSample
set.seed(1010)
down_inside <- train(x = trainData[,-16], y = trainData$smoking, method = "rf", ntree=100,
                     metric = "ROC", trControl = ctrl)

# upSample
ctrl$sampling <- "up"
set.seed(1010)
up_inside <- train(x = trainData[,-16], y = trainData$smoking, method = "rf", ntree=100,
                   metric = "ROC", trControl = ctrl)

# ROSE
ctrl$sampling <- "rose"
set.seed(1010)
rose_inside <- train(x = trainData[,-16], y = trainData$smoking, method = "rf", ntree=100,
                     metric = "ROC", trControl = ctrl)

# originalan model bez uzrokovanja
ctrl$sampling <- NULL
set.seed(1010)
orig_fit <- train(x = trainData[,-16], y = trainData$smoking, method = "rf", ntree=100,
                  metric = "ROC", trControl = ctrl)

# pravimo listu sa svim modelima i uporedjujemo ih
inside_models <- list(original = orig_fit,
                      down = down_inside,
                      up = up_inside,
                      ROSE = rose_inside)
inside_resampling <- resamples(inside_models)
summary(inside_resampling, metric = "ROC")
# ROC 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# original 0.8642660 0.8689876 0.8690381 0.8700769 0.8715399 0.8765531    0
# down     0.8539793 0.8572228 0.8581053 0.8592855 0.8597733 0.8673468    0
# up       0.8666077 0.8687449 0.8701872 0.8711191 0.8703805 0.8796750    0
# ROSE     0.8049510 0.8064405 0.8132547 0.8110880 0.8150062 0.8157874    0

# najbolji je upSample

# kreiranje predikcije, matrice konfuzije i evalucionih metrika
rf2.pred <- predict(up_inside$finalModel, newdata = testData, type = "class")

rf2.cm <- table(true = testData$smoking, predicted = rf2.pred)
rf2.cm
#  true    No  Yes
#   No    3932 1001
#   Yes    706 2157

eval.rf2 <- getEvaluationMetrics(rf2.cm)
eval.rf2
# Accuracy Precision    Recall        F1 
# 0.7810416 0.6830272 0.7534055 0.7164923

# KREIRANJE TRECEG MODELA

m<- sqrt(ncol(data))
grid <- expand.grid(mtry = c(m-2,m-1,m,m+1,m+2))

train_control <- trainControl(method = "cv",
                              number = 5,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

model_rf <- train(x = trainData[,-16], y = trainData$smoking, method = "rf", metric = "ROC", ntree=100,
                  tuneGrid = grid, trControl = train_control)
best_mtry <- model_rf$bestTune$mtry
# best_mtry je 3
model_rf$finalModel
# Confusion matrix:
#         No Yes class.error
# No  16553 3180   0.1611514
# Yes  3232 8223   0.2821475

# kreiranje predikcije, matrice konfuzije i evalucionih metrika
rf3 <- randomForest(smoking ~., data = trainData, mtry = best_mtry) 
rf3.pred <- predict(object = rf3, newdata = testData, type = "class")

rf3.cm <- table(true = testData$smoking, predicted = rf3.pred)
rf3.cm
#       predicted
# true    No  Yes
#   No  4107  826
#   Yes  810 2053
eval.rf3 <- getEvaluationMetrics(rf3.cm)
eval.rf3
# Accuracy Precision    Recall        F1 
# 0.7901488 0.7130948 0.7170800 0.7150819 


# uporedjivanje rezultata modela i tumacenje
data.frame(rbind(eval.rf1,eval.rf2,eval.rf3), row.names = c("Prvi model","Drugi model","Treci model"))

#               Accuracy Precision    Recall        F1
# Prvi model  0.7887378 0.7097999 0.7184771 0.7141121
# Drugi model 0.7810416 0.6830272 0.7534055 0.7164923
# Treci model 0.7901488 0.7130948 0.7170800 0.7150819

# treci model daje nam najbolje rezultate poredjenjem rezultata svakog modela

# znacajnost varijabli za treci model

png("varImpRF.png", width = 1200, height = 800, res = 150)
randomForest::varImpPlot(rf3, main="Variable Importance Plot")
dev.off()

varImp(rf3) |> as.data.frame() |> dplyr::arrange(desc(Overall))
#                     Overall
# hemoglobin       2070.42227
# Gtp              1872.88847
# height.cm.       1618.94376
# triglyceride     1478.80056
# waist.cm.        1186.92997
# HDL              1161.83485
# systolic         1066.69697
# relaxation       1001.27423
# serum.creatinine  861.65400
# weight.kg.        855.35450
# age               831.29556
# dental.caries     154.35040
# Urine.protein      85.98348
# hearing.right.     39.25982
# hearing.left.      37.21894















