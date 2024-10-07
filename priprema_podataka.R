# ucitavamo izabrani dataset za rad
data <- read.csv("train_dataset.csv", stringsAsFactors = FALSE )
str(data)

# proveravamo da li imamo nedostajajuce vrednosti
apply(data,MARGIN = 2,FUN = function(x) sum(is.na(x)))
apply(data,MARGIN = 2,FUN = function(x) sum(x==""))
apply(data,MARGIN = 2,FUN = function(x) sum(x==" "))
apply(data,MARGIN = 2,FUN = function(x) sum(x=="-"))
# zakljucujemo da ne postoji nijedna nedostajajuca vrednost

# numericke varijbale binarnog tipa 0,1 cemo pretvoriti u faktorske, 
# takodje i varijablu Urine proteine jer je data u vidu odredjenih nivoa 
data$hearing.left.<-as.factor(data$hearing.left.)
data$hearing.right.<-as.factor(data$hearing.right.)
data$Urine.protein<-as.factor(data$Urine.protein)
data$dental.caries<-as.factor(data$dental.caries)
data$smoking<-as.factor(data$smoking)

# sledi ispitivanje znacajnosti varijabli koriscenjem plotova
library(ggplot2)
# numericke varijable
ggplot(data = data, aes(x = age, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = height.cm., fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = weight.kg., fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = waist.cm., fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = eyesight.left., fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = eyesight.right., fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = systolic, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = relaxation, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = fasting.blood.sugar, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = Cholesterol, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = triglyceride, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = HDL, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = LDL, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = hemoglobin, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = serum.creatinine, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = ALT, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = AST, fill = smoking)) + geom_density(alpha=0.6)
ggplot(data = data, aes(x = Gtp, fill = smoking)) + geom_density(alpha=0.6)

# varijable age, height.cm., weight.kg.,waist.cm., systolic, relaxation, triglyceride, hemoglobin
# imaju znacajna ostupanja u odnosu na svoje vrednosti za vrednosti izlazne varijable smoking

# kako se broj godina povecava tako opada sansa da je osoba pusac
# povecanje visine, kilaze ili obima struka ukazuje na vecu verovatnocu da je osoba pusac
# varijabla koja govori o donjem pritisku osobe pokazuje manja znacajnija ostupanja, ukoliko je pritisak nizi manja je sansa 
# da govorimo o pusacu, pa proporcionalno i poviseni pritisak aludira pre na pusaca
# veci postotak hemoglobina ili triglicerida proporcionalno ukazuje na veci procenat sanse da je osoba pusac

# varijable HDL, serum.creatinine, Gtp pokazuju minimalne razlike u predvidjaju vrednosti izlazne varijable smoking, 
# ali ih za sad ostavljamo kao znacajne

# kod varijabli eyesight.left, eyesight.right, fasting.blood.sugar, Cholesterol, LDL, AST i ALT vrednosti
# izlazne promenljive smoking u istom procentu su zastupljene za sve vrednosti prethodno pomenutih varijabli
# i zato pomenute odstranjujemo iz naseg predvidjanja, jer ne mogu uticati znacajno na isto
data$eyesight.left.<-NULL
data$eyesight.right.<-NULL
data$fasting.blood.sugar<-NULL
data$Cholesterol<-NULL
data$LDL<-NULL
data$ALT<-NULL
data$AST<-NULL

#faktorske varijable
ggplot(data = data, aes(x = hearing.left., fill = smoking)) + geom_bar(position="fill")
ggplot(data = data, aes(x = hearing.right., fill = smoking)) + geom_bar(position="fill")
ggplot(data = data, aes(x = Urine.protein, fill = smoking)) + geom_bar(position="fill")
ggplot(data = data, aes(x = dental.caries, fill = smoking)) + geom_bar(position="fill")
# sve faktorske varijable imaju minimalne razlike u rezultatima izlazne promenljive,
# ali cemo ih za sada ostaviti kao znacajne za nase predvidjanje

saveRDS(data,file = "preprocessed_dataset.RDS")




