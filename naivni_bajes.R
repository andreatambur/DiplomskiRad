# ucitavanje sredjenog dataset-a
preprocessed_dataset<- readRDS("preprocessed_dataset.RDS")
data <- preprocessed_dataset

# ucitavanje funkcije
source("funkcija.R")

# instaliranje paketa i ucitavanje biblioteka neophodnih za rad
install.packages("bnlearn")
install.packages('caret')
install.packages("naivebayes")
#install.packages("pROC")
library(caret)
library(bnlearn)
library(e1071)
library(pROC)
library(rpart)
library(ROSE)
library(naivebayes)

str(data)

# dodatno sredjivanje podataka
# provera da li numericke varijable imaju normalnu raspodelu
apply(data[ ,c(1:4,7:11,13,14)], 2, function(x) shapiro.test(sample(x,size=5000)))

# sve promenljive kojima je p-value<0,05 nemaju normalnu raspodelu i njih diskretizujemo, u nasem slucaju sve
# pre toga potrebno je da sve integer varijable prebacimo u numeric
data$age<-as.numeric(data$age)
data$height.cm.<-as.numeric(data$height.cm.)
data$weight.kg.<-as.numeric(data$weight.kg.)
data$systolic<-as.numeric(data$systolic)
data$relaxation<-as.numeric(data$relaxation)
data$triglyceride<-as.numeric(data$triglyceride)
data$HDL<-as.numeric(data$HDL)
data$Gtp<-as.numeric(data$Gtp)

discretized <- discretize(data = data[,c(1:4,7:11,13,14)], method = "quantile", breaks = c(2,3,4,5,5,5,5,5,5,5,5))
summary(discretized)

# krairanje novog data seta od diskretizovanih podataka i faktorskih varijabli
newData<-as.data.frame(cbind(discretized,data[,c(5,6,12,15,16)]))
data<-newData

# kreiranje trening i test setova
set.seed(1010) 
i <- createDataPartition(data$smoking, p = 0.8, list = FALSE)
trainData<- data[i, ]
testData <- data[-i, ]

# KREIRANJE PRVOG MODELA

# kreiranje modela, predikcije i matrice konfuzije
nb1 <- naiveBayes(smoking ~., data = trainData)
nb1.pred <- predict(nb1, newdata = testData, type = "class")

nb1.cm <- table(true = testData$smoking, predicted = nb1.pred)
nb1.cm
#        predicted
# true    0    1
# 0     3315 1618
# 1      762 2101

# evalucione metrike
eval.nb1 <- getEvaluationMetrics(nb1.cm)
eval.nb1
# Accuracy Precision    Recall        F1 
# 0.6947152 0.5649368 0.7338456 0.6384078

# KREIRANJE DRUGOG MODDELA

nb2.pred.prob <- predict(nb1, newdata = testData, type = "raw")
nb2.pred.prob
# za response izlazna varijabla treba da bude integer , za predictor vrednost dajemo verovatnocu pozitivne klase,
# levels je uredjen tako da prvo ide negativna, pa pozitivna klasa
nb2.roc <- roc(response = as.integer(testData$smoking),
               predictor = nb2.pred.prob[,2],
               levels = c(1,2))
# sensitivity odgovara recallu: u odnosu na sve osobe koje su pusaci (klase koje su pozitivne),
# koji je udeo onih koje smo mi predvideli da su pusaci (da jesu pozitivne)
# specificity je isto samo se odnosi na negativnu klasu: u odnosu na sve osobe koje nisu pusaci
# (sve klase koje su negativne), koji je udeo onih koje smo mi predvideli da nisu pusaci (da jesu negativne)
plot.roc(nb2.roc)

nb2.roc$auc
#Area under the curve: 0.7727
# sto je AUC - area under the curve veca, to se klasifikator smatra boljim
# ako je 1 onda moze perfektno da razlikuje koja je pozitivna, a koja negativna klasa
# ako je 0.5 onda ne moze da razlikuje, to je najgora situacija
# u nasem slucaju mi imamo 0.7727 sanse da razlikujemo pozitivnu od negativne klase


plot.roc(nb2.roc, print.thres = TRUE, print.thres.best.method = "youden")
nb2.coords <- coords(nb2.roc, ret = c("accuracy","spec","sens","threshold"),
                     x = "best",best.method = "youden")
nb2.coords
#              accuracy specificity sensitivity threshold
# threshold 0.6791945   0.5647679   0.8763535 0.1409149

threshold <- nb2.coords[1,'threshold']
threshold
# njegova vrednost je 0.1409149

nb2.pred <- ifelse(test = nb2.pred.prob[,2]>= threshold,
                   yes = "1", no = "0")

nb2.cm<- table(true = testData$smoking, predicted = nb2.pred)
nb2.cm
#        predicted
# true    0    1
# 0     2786 2147
# 1      354 2509

eval.nb2 <- getEvaluationMetrics(nb2.cm)
eval.nb2
# Accuracy Precision    Recall        F1 
# 0.6791945 0.5388746 0.8763535 0.6673760 

# KREIRANJE TRECEG MODELA

# kreiranje modela koriscenjem balansiranog dataset-a
trainData$smoking <- factor(trainData$smoking, levels = c(0, 1), labels = c("No", "Yes"))
testData$smoking <- factor(testData$smoking, levels = c(0, 1),labels = c("No", "Yes"))
grid <- expand.grid(laplace = 1, usekernel= FALSE, adjust = 1)

# kreiranje kontrolnog modela koji koristi metod unakrsne krosvalidacije s ponavljanjem radi vece pouzdanosti
ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down")
# downSample
set.seed(1010)
down_inside <- train(x = trainData[,-16], y = trainData$smoking, method = "naive_bayes",
                     metric = "ROC", trControl = ctrl, tuneGrid = grid)


# upSample
ctrl$sampling <- "up"
set.seed(1010)
up_inside <- train(x = trainData[,-16], y = trainData$smoking, method = "naive_bayes",
                   metric = "ROC", trControl = ctrl, tuneGrid = grid)

# ROSE
ctrl$sampling <- "rose"
set.seed(1010)
rose_inside <- train(x = trainData[,-16], y = trainData$smoking, method = "naive_bayes",
                     metric = "ROC", trControl = ctrl, tyuneGrid = grid)

# originalan model bez uzrokovanja
ctrl$sampling <- NULL
set.seed(1010)
orig_fit <- train(x = trainData[,-16], y = trainData$smoking, method = "naive_bayes",
                  metric = "ROC", trControl = ctrl,tuneGrid = grid)

# pravimo listu sa svim modelima i uporedjujemo ih
inside_models <- list(original = orig_fit,
                      down = down_inside,
                      up = up_inside,
                      ROSE = rose_inside)
inside_resampling <- resamples(inside_models)
summary(inside_resampling, metric = "ROC")

# ROC 
#                Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# original 0.7665196 0.7774733 0.7833755 0.7826849 0.7881930 0.8026115    0
# down     0.7646856 0.7777378 0.7835097 0.7826875 0.7868684 0.8000530    0
# up       0.7666073 0.7776639 0.7834567 0.7826839 0.7881429 0.8021850    0
# ROSE     0.7667060 0.7774130 0.7831071 0.7825914 0.7881833 0.8031720    0

# svi modeli imaju skoro pa iste rezultate, uz sasvim zanemarljiva odstupanja jednog od drugog
# koristicemo up_inside za treci model

# kreiranje predikcije, matrice konfuzije i evalucionih metrika
nb3.pred <- predict(up_inside$finalModel, newdata = testData, type = "class")

nb3.cm <- table(true = testData$smoking, predicted = nb3.pred)
nb3.cm
#       predicted
# true    No  Yes
# No     3136 1797
# Yes     617 2246

eval.nb3 <- getEvaluationMetrics(nb3.cm)
eval.nb3
# Accuracy Precision    Recall        F1 
# 0.6903540 0.5555281 0.7844918 0.6504489 


# uporedjivanje rezultata modela i tumacenje
data.frame(rbind(eval.nb1, eval.nb2, eval.nb3), row.names = c("Prvi model","Drugi model","Treci model"))
 
#              Accuracy Precision    Recall        F1
# Prvi model  0.6947152 0.5649368 0.7338456 0.6384078
# Drugi model 0.6791945 0.5388746 0.8763535 0.6673760
# Treci model 0.6903540 0.5555281 0.7844918 0.6504489



















