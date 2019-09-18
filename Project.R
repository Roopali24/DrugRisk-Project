rm(list=ls())
setwd("~/Box Sync/IST 597")
url="https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"
df=read.csv(url, sep=",", header=FALSE)[,c(2:13,19,21,23,27,29)]
summary(df)
dat=cbind(df[,1:12], 1*(df$V19=="CL0" | df$V19=="CL1"), 1*(df$V21=="CL0" | df$V21=="CL1"),
         1*(df$V23=="CL0" | df$V23=="CL1"), 1*(df$V27=="CL0" | df$V27=="CL1"), 1*(df$V29=="CL0" | df$V29=="CL1"))
colnames(dat)=c(colnames(df[,1:12]),"Cannabis","Coke","Ecstasy","LSD","Mushrooms")
summary(dat)

## EDA
drug_user=1*(dat$Cannabis==1 | dat$Coke==1 | dat$Ecstasy ==1 | dat$LSD==1 | dat$Mushrooms ==1)

## Proportion of people who do drugs in different classes
## Age
drug_per=rep(NA,6)
drug_per[1]=sum(drug_user[dat$V2== -0.95197])/length(drug_user[dat$V2== -0.95197])
drug_per[2]=sum(drug_user[dat$V2== -0.07854])/length(drug_user[dat$V2== -0.07854])
drug_per[3]=sum(drug_user[dat$V2== 0.49788])/length(drug_user[dat$V2== 0.49788])
drug_per[4]=sum(drug_user[dat$V2== 1.09449])/length(drug_user[dat$V2== 1.09449])
drug_per[5]=sum(drug_user[dat$V2== 1.82213])/length(drug_user[dat$V2== 1.82213])
drug_per[6]=sum(drug_user[dat$V2== 2.59171])/length(drug_user[dat$V2== 2.59171])

## Ethnicity (White vs non white)
drug_per=rep(NA,2)
drug_per[1]=sum(drug_user[dat$V6== -0.31685])/length(drug_user[dat$V6== -0.31685])
drug_per[2]=sum(drug_user[dat$V6!= -0.31685])/length(drug_user[dat$V6 != -0.31685])

## Gender
drug_per=rep(NA,2)
drug_per[1]=sum(drug_user[dat$V3== 0.48246])/length(drug_user[dat$V3== 0.48246])
drug_per[2]=sum(drug_user[dat$V3== -0.48246])/length(drug_user[dat$V3== -0.48246])

## Disparate Impact
table(dat$Cannabis,dat$V3) #gender
(527/(527+415))/(738/(738+205)) # for female minority
table(dat$Cannabis,dat$V6)
(10/(10+23))/(1165/(555+1165)) #for black minority
(7/(7+19))/(1165/(555+1165)) # asian
((14+16+3)/(14+6+16+4+3))/(1165/(555+1165)) #mixed
((14+16+3+10+7+50+3)/(14+6+16+4+3+50+13+7+19+10+23))/(1165/(555+1165)) # White vs Non-white

table(dat$Coke,dat$V3) #gender
(270/(672+270))/(417/(417+526)) # for female minority
table(dat$Coke,dat$V6)
(4/(4+29))/(631/(631+1089)) #for black minority
(3/(3+23))/(631/(631+1089)) # asian
((6+11+1)/(6+14+11+9+1+2))/(631/(631+1089)) #mixed
((6+11+1+31+4+3)/(6+14+11+9+1+2+31+32+4+29+3+23))/(631/(631+1089)) # White vs Non-white

table(dat$Ecstasy,dat$V3) #gender
(268/(674+268))/(483/(483+460)) # for female minority
table(dat$Ecstasy,dat$V6)
(4/(4+29))/(689/(689+1031)) #for black minority
(6/(6+20))/(689/(689+1031)) # asian
((8+10)/(8+10+12+10+3))/(689/(689+1031)) #mixed
((8+10+4+6+34)/(8+10+12+10+3+4+29+6+20+34+29))/(689/(689+1031)) # White vs Non-white

table(dat$LSD,dat$V3) #gender
(165/(165+777))/(392/(392+551)) # for female minority
table(dat$LSD,dat$V6)
(1/(1+32))/(505/(505+1215)) #for black minority
(3/(3+23))/(505/(505+1215)) # asian
((6+10+2)/(6+14+10+10+2+1))/(505/(505+1215)) #mixed
((6+10+2+1+3+30)/(6+14+10+10+2+1+1+32+3+23+30+33))/(505/(505+1215)) # White vs Non-white

table(dat$Mushrooms,dat$V3) #gender
(231/(231+711))/(463/(463+480)) # for female minority
table(dat$Mushrooms,dat$V6)
(3/(3+30))/(635/(635+1085)) #for black minority
(4/(4+22))/(635/(635+1085)) # asian
((8+11+2)/(8+12+11+9+2+1))/(635/(635+1085)) #mixed
((8+11+2+3+4+31)/(8+12+11+9+2+1+3+30+4+22+31+32))/(635/(635+1085))

## PCA
pca=prcomp(scale(dat[,1:12]))
summary(pca)
par(mfrow=c(1,1))
plot(pca,type='l')

## correlation
cor(dat[,1:12]) # none are highly correlated

## Training and testing data. 70% training 30% testing
set.seed(123)
n=round(nrow(dat)*.7)
train = sample(1:nrow(dat), size=n)
df.train=dat[train,]
df.test=dat[-train,]

## Logistic regression
library(ggplot2)
library(boot)
library(AUC)
library(caret)

## For Cannabis
dat1=df.train[,1:13]
log.fit1=glm(as.factor(Cannabis) ~.,data=dat1,family="binomial")
step(log.fit1)
#log.fit1=glm(formula = as.factor(Cannabis) ~ V2 + V3 + V4 + V5 + V9 + 
#               V11 + V13, family = "binomial", data = dat1)
cv.glm(dat1,log.fit1,K=10)$delta
summary(log.fit1)

log_train=predict(log.fit1, dat1,type = "response")
RC=roc(log_train,as.factor(dat1$Cannabis))
auc(RC)
log_train[log_train>0.5]=1
log_train[log_train<0.5]=0
precision(table(factor(log_train,c(0,1)),factor(dat1$Cannabis)), relevant='1')
recall(table(factor(log_train,c(0,1)),factor(dat1$Cannabis)), relevant='1')

log_test=predict(log.fit1, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$Cannabis))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
1-sum((log_test-df.test$Cannabis)^2)/nrow(df.test)

## For Coke
dat2=df.train[,c(1:12,14)]
log.fit2=glm(Coke ~.,data=dat2,family="binomial")
step(log.fit2)
#log.fit2=glm(formula = Coke ~ V2 + V5 + V6 + V7 + V8 + V10 + V11 + V13, 
 #            family = "binomial", data = dat2)
cv.glm(dat2,log.fit2,K=10)$delta
summary(log.fit2)

log_train=predict(log.fit2, dat2,type = "response")
RC=roc(log_train,as.factor(dat2$Coke))
auc(RC)
log_train[log_train>0.5]=1
log_train[log_train<0.5]=0
precision(table(factor(log_train,c(0,1)),factor(dat2$Coke)), relevant='1')
recall(table(factor(log_train,c(0,1)),factor(dat2$Coke)), relevant='1')

log_test=predict(log.fit2, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$Coke))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Coke)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Coke)), relevant='1')
1-sum((log_test-df.test$Coke)^2)/nrow(df.test)

## For Ecstasy
dat3=df.train[,c(1:12,15)]
log.fit3=glm(Ecstasy ~.,data=dat3,family="binomial")
step(log.fit3)
#log.fit3=glm(formula = Ecstasy ~ V2 + V3 + V4 + V5 + V9 + V11 + V12 + 
 #              V13, family = "binomial", data = dat3)
cv.glm(dat3,log.fit3,K=10)$delta
summary(log.fit3)

log_train=predict(log.fit3, dat3,type = "response")
RC=roc(log_train,as.factor(dat3$Ecstasy))
auc(RC)
log_train[log_train>0.5]=1
log_train[log_train<0.5]=0
precision(table(factor(log_train,c(0,1)),factor(dat3$Ecstasy)), relevant='1')
recall(table(factor(log_train,c(0,1)),factor(dat3$Ecstasy)), relevant='1')

log_test=predict(log.fit3, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$Ecstasy))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
1-sum((log_test-df.test$Ecstasy)^2)/nrow(df.test)


## For LSD
dat4=df.train[,c(1:12,16)]
log.fit4=glm(LSD ~.,data=dat4,family="binomial")
cv.glm(dat4,log.fit4,K=10)$delta
summary(log.fit4)

log_train=predict(log.fit4, dat4,type = "response")
RC=roc(log_train,as.factor(dat4$LSD))
auc(RC)
log_train[log_train>0.5]=1
log_train[log_train<0.5]=0
precision(table(factor(log_train,c(0,1)),factor(dat4$LSD)), relevant='1')
recall(table(factor(log_train,c(0,1)),factor(dat4$LSD)), relevant='1')

log_test=predict(log.fit4, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$LSD))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$LSD)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$LSD)), relevant='1')
1-sum((log_test-df.test$LSD)^2)/nrow(df.test)


## For Mushrooms
dat5=df.train[,c(1:12,17)]
log.fit5=glm(Mushrooms ~.,data=dat5,family="binomial")
cv.glm(dat5,log.fit5,K=10)$delta
summary(log.fit5)

log_train=predict(log.fit5, dat5,type = "response")
RC=roc(log_train,as.factor(dat5$Mushrooms))
auc(RC)
log_train[log_train>0.5]=1
log_train[log_train<0.5]=0
precision(table(factor(log_train,c(0,1)),factor(dat5$Mushrooms)), relevant='1')
recall(table(factor(log_train,c(0,1)),factor(dat5$Mushrooms)), relevant='1')

log_test=predict(log.fit5, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$Mushrooms))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
1-sum((log_test-df.test$Mushrooms)^2)/nrow(df.test)


## Decision Trees
library(rpart)

## For Cannabis
dat1=df.train[,1:13]
tree.fit1=rpart(factor(Cannabis) ~.,data=dat1, control=rpart.control(cp=0))
bestcp <- tree.fit1$cptable[which.min(tree.fit1$cptable[,"xerror"]),"CP"] 
tree.fit1=rpart(factor(Cannabis) ~.,data=dat1, control=rpart.control(cp=bestcp))
rpart.plot(tree.fit1)

tree_train = predict(tree.fit1,dat1,type="prob")
RC=roc(tree_train[,2],as.factor(dat1$Cannabis))
auc(RC)
tree_train=1*(tree_train[,2]>0.5)
precision(table(factor(tree_train,c(0,1)),factor(dat1$Cannabis)), relevant='1')
recall(table(factor(tree_train,c(0,1)),factor(dat1$Cannabis)), relevant='1')

tree_test=predict(tree.fit1, newdata=df.test[,1:12], type="prob")
RC=roc(tree_test[,2],factor(df.test$Cannabis))
auc(RC)
tree_test=1*(tree_test[,2]>0.5)
precision(table(factor(tree_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
recall(table(factor(tree_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
1-sum((tree_test-df.test$Cannabis)^2)/nrow(df.test)

## For Coke
dat2=df.train[,c(1:12,14)]
tree.fit2=rpart(factor(Coke) ~.,data=dat2, control=rpart.control(cp=0))
bestcp <- tree.fit2$cptable[which.min(tree.fit2$cptable[,"xerror"]),"CP"] 
tree.fit2=rpart(factor(Coke) ~.,data=dat2, control=rpart.control(cp=bestcp))
rpart.plot(tree.fit2)

tree_train = predict(tree.fit2,dat2,type="prob")
RC=roc(tree_train[,2],as.factor(dat2$Coke))
auc(RC)
tree_train=1*(tree_train[,2]>0.5)
precision(table(factor(tree_train,c(0,1)),factor(dat2$Coke)), relevant='1')
recall(table(factor(tree_train,c(0,1)),factor(dat2$Coke)), relevant='1')

tree_test=predict(tree.fit2, newdata=df.test[,1:12], type="prob")
RC=roc(tree_test[,2],factor(df.test$Coke))
auc(RC)
tree_test=1*(tree_test[,2]>0.5)
precision(table(factor(tree_test,c(0,1)),factor(df.test$Coke)), relevant='1')
recall(table(factor(tree_test,c(0,1)),factor(df.test$Coke)), relevant='1')
1-sum((tree_test-df.test$Coke)^2)/nrow(df.test)

## For Ecstasy
dat3=df.train[,c(1:12,15)]
tree.fit3=rpart(factor(Ecstasy) ~.,data=dat3, control=rpart.control(cp=0))
bestcp <- tree.fit3$cptable[which.min(tree.fit3$cptable[,"xerror"]),"CP"] 
tree.fit3=rpart(factor(Ecstasy) ~.,data=dat3, control=rpart.control(cp=bestcp))
rpart.plot(tree.fit3)

tree_train = predict(tree.fit3,dat3,type="prob")
RC=roc(tree_train[,2],as.factor(dat3$Ecstasy))
auc(RC)
tree_train=1*(tree_train[,2]>0.5)
precision(table(factor(tree_train,c(0,1)),factor(dat3$Ecstasy)), relevant='1')
recall(table(factor(tree_train,c(0,1)),factor(dat3$Ecstasy)), relevant='1')

tree_test=predict(tree.fit3, newdata=df.test[,1:12], type="prob")
RC=roc(tree_test[,2],factor(df.test$Ecstasy))
auc(RC)
tree_test=1*(tree_test[,2]>0.5)
precision(table(factor(tree_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
recall(table(factor(tree_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
1-sum((tree_test-df.test$Ecstasy)^2)/nrow(df.test)


## For LSD
dat4=df.train[,c(1:12,16)]
tree.fit4=rpart(factor(LSD) ~.,data=dat4, control=rpart.control(cp=0))
bestcp <- tree.fit4$cptable[which.min(tree.fit4$cptable[,"xerror"]),"CP"] 
tree.fit4=rpart(factor(LSD) ~.,data=dat4, control=rpart.control(cp=bestcp))
rpart.plot(tree.fit4)

tree_train = predict(tree.fit4,dat4,type="prob")
RC=roc(tree_train[,2],as.factor(dat4$LSD))
auc(RC)
tree_train=1*(tree_train[,2]>0.5)
precision(table(factor(tree_train,c(0,1)),factor(dat4$LSD)), relevant='1')
recall(table(factor(tree_train,c(0,1)),factor(dat4$LSD)), relevant='1')

tree_test=predict(tree.fit4, newdata=df.test[,1:12], type="prob")
RC=roc(tree_test[,2],factor(df.test$LSD))
auc(RC)
tree_test=1*(tree_test[,2]>0.5)
precision(table(factor(tree_test,c(0,1)),factor(df.test$LSD)), relevant='1')
recall(table(factor(tree_test,c(0,1)),factor(df.test$LSD)), relevant='1')
1-sum((tree_test-df.test$LSD)^2)/nrow(df.test)


## For Mushrooms
dat5=df.train[,c(1:12,17)]
tree.fit5=rpart(factor(Mushrooms) ~.,data=dat5, control=rpart.control(cp=0))
bestcp <- tree.fit5$cptable[which.min(tree.fit5$cptable[,"xerror"]),"CP"] 
tree.fit5=rpart(factor(Mushrooms) ~.,data=dat5, control=rpart.control(cp=bestcp))
rpart.plot(tree.fit5)

tree_train = predict(tree.fit5,dat5,type="prob")
RC=roc(tree_train[,2],as.factor(dat5$Mushrooms))
auc(RC)
tree_train=1*(tree_train[,2]>0.5)
precision(table(factor(tree_train,c(0,1)),factor(dat5$Mushrooms)), relevant='1')
recall(table(factor(tree_train,c(0,1)),factor(dat5$Mushrooms)), relevant='1')

tree_test=predict(tree.fit5, newdata=df.test[,1:12], type="prob")
RC=roc(tree_test[,2],factor(df.test$Mushrooms))
auc(RC)
tree_test=1*(tree_test[,2]>0.5)
precision(table(factor(tree_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
recall(table(factor(tree_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
1-sum((tree_test-df.test$Mushrooms)^2)/nrow(df.test)


## Naive Bayes
library(e1071)

## For Cannabis
dat1=df.train[,1:13]
nb.fit1=naiveBayes(as.factor(Cannabis) ~.,data=dat1)

#nb_train=predict(nb.fit1, dat1, type="raw")
#RC=roc(nb_train[,2],as.factor(dat1$Cannabis))
#auc(RC)
#nb_train=predict(nb.fit1, dat1, type="class")
#precision(table(factor(nb_train,c(0,1)),factor(dat1$Cannabis)), relevant='1')
#recall(table(factor(nb_train,c(0,1)),factor(dat1$Cannabis)), relevant='1')

nb_test=predict(nb.fit1, newdata=df.test[,1:12], type="raw")
RC=roc(nb_test[,2],factor(df.test$Cannabis))
auc(RC)
nb_test=1*(nb_test[,2]>0.5)
precision(table(factor(nb_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
recall(table(factor(nb_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
1-sum((nb_test-df.test$Cannabis)^2)/nrow(df.test)

## For Coke
dat2=df.train[,c(1:12,14)]
nb.fit2=naiveBayes(Coke ~.,data=dat2)

nb_test=predict(nb.fit2, newdata=df.test[,1:12], type="raw")
RC=roc(nb_test[,2],factor(df.test$Coke))
auc(RC)
nb_test=1*(nb_test[,2]>0.5)
precision(table(factor(nb_test,c(0,1)),factor(df.test$Coke)), relevant='1')
recall(table(factor(nb_test,c(0,1)),factor(df.test$Coke)), relevant='1')
1-sum((nb_test-df.test$Coke)^2)/nrow(df.test)

## For Ecstasy
dat3=df.train[,c(1:12,15)]
nb.fit3=naiveBayes(Ecstasy ~.,data=dat3)

nb_test=predict(nb.fit3, newdata=df.test[,1:12], type="raw")
RC=roc(nb_test[,2],factor(df.test$Ecstasy))
auc(RC)
nb_test=1*(nb_test[,2]>0.5)
precision(table(factor(nb_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
recall(table(factor(nb_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
1-sum((nb_test-df.test$Ecstasy)^2)/nrow(df.test)

## For LSD
dat4=df.train[,c(1:12,16)]
nb.fit4=naiveBayes(LSD ~.,data=dat4)

nb_test=predict(nb.fit4, newdata=df.test[,1:12], type="raw")
RC=roc(nb_test[,2],factor(df.test$LSD))
auc(RC)
nb_test=1*(nb_test[,2]>0.5)
precision(table(factor(nb_test,c(0,1)),factor(df.test$LSD)), relevant='1')
recall(table(factor(nb_test,c(0,1)),factor(df.test$LSD)), relevant='1')
1-sum((nb_test-df.test$LSD)^2)/nrow(df.test)

## For Mushrooms
dat5=df.train[,c(1:12,17)]
nb.fit5=naiveBayes(Mushrooms ~.,data=dat5)

nb_test=predict(nb.fit5, newdata=df.test[,1:12], type="raw")
RC=roc(nb_test[,2],factor(df.test$Mushrooms))
auc(RC)
nb_test=1*(nb_test[,2]>0.5)
precision(table(factor(nb_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
recall(table(factor(nb_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
1-sum((nb_test-df.test$Mushrooms)^2)/nrow(df.test)


##### Without gender and race ######
df.train=dat[train,-c(2,5)]
df.test=dat[-train,-c(2,5)]

## Logistic regression
## For Cannabis
dat1=df.train[,1:11]
log.fit1=glm(as.factor(Cannabis) ~.,data=dat1,family="binomial")
summary(log.fit1)

log_test=predict(log.fit1, newdata=df.test[,1:10], type="response")
RC=roc(log_test,factor(df.test$Cannabis))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
1-sum((log_test-df.test$Cannabis)^2)/nrow(df.test)

## For Coke
dat2=df.train[,c(1:10,12)]
log.fit2=glm(Coke ~.,data=dat2,family="binomial")
summary(log.fit2)

log_test=predict(log.fit2, newdata=df.test[,1:10], type="response")
RC=roc(log_test,factor(df.test$Coke))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Coke)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Coke)), relevant='1')
1-sum((log_test-df.test$Coke)^2)/nrow(df.test)

## For Ecstasy
dat3=df.train[,c(1:10,13)]
log.fit3=glm(Ecstasy ~.,data=dat3,family="binomial")
summary(log.fit3)

log_test=predict(log.fit3, newdata=df.test[,1:10], type="response")
RC=roc(log_test,factor(df.test$Ecstasy))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
1-sum((log_test-df.test$Ecstasy)^2)/nrow(df.test)


## For LSD
dat4=df.train[,c(1:10,14)]
log.fit4=glm(LSD ~.,data=dat4,family="binomial")

log_test=predict(log.fit4, newdata=df.test[,1:10], type="response")
RC=roc(log_test,factor(df.test$LSD))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$LSD)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$LSD)), relevant='1')
1-sum((log_test-df.test$LSD)^2)/nrow(df.test)


## For Mushrooms
dat5=df.train[,c(1:10,15)]
log.fit5=glm(Mushrooms ~.,data=dat5,family="binomial")

log_test=predict(log.fit5, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$Mushrooms))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
1-sum((log_test-df.test$Mushrooms)^2)/nrow(df.test)

## Naive Bayes

## For Cannabis
nb.fit1=naiveBayes(as.factor(Cannabis) ~.,data=dat1)

nb_test=predict(nb.fit1, newdata=df.test[,1:10], type="raw")
RC=roc(nb_test[,2],factor(df.test$Cannabis))
auc(RC)
nb_test=1*(nb_test[,2]>0.5)
precision(table(factor(nb_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
recall(table(factor(nb_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
1-sum((nb_test-df.test$Cannabis)^2)/nrow(df.test)

## For Coke
nb.fit2=naiveBayes(Coke ~.,data=dat2)

nb_test=predict(nb.fit2, newdata=df.test[,1:10], type="raw")
RC=roc(nb_test[,2],factor(df.test$Coke))
auc(RC)
nb_test=1*(nb_test[,2]>0.5)
precision(table(factor(nb_test,c(0,1)),factor(df.test$Coke)), relevant='1')
recall(table(factor(nb_test,c(0,1)),factor(df.test$Coke)), relevant='1')
1-sum((nb_test-df.test$Coke)^2)/nrow(df.test)

## For Ecstasy
nb.fit3=naiveBayes(Ecstasy ~.,data=dat3)

nb_test=predict(nb.fit3, newdata=df.test[,1:10], type="raw")
RC=roc(nb_test[,2],factor(df.test$Ecstasy))
auc(RC)
nb_test=1*(nb_test[,2]>0.5)
precision(table(factor(nb_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
recall(table(factor(nb_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
1-sum((nb_test-df.test$Ecstasy)^2)/nrow(df.test)

## For LSD
nb.fit4=naiveBayes(LSD ~.,data=dat4)

nb_test=predict(nb.fit4, newdata=df.test[,1:10], type="raw")
RC=roc(nb_test[,2],factor(df.test$LSD))
auc(RC)
nb_test=1*(nb_test[,2]>0.5)
precision(table(factor(nb_test,c(0,1)),factor(df.test$LSD)), relevant='1')
recall(table(factor(nb_test,c(0,1)),factor(df.test$LSD)), relevant='1')
1-sum((nb_test-df.test$LSD)^2)/nrow(df.test)

## For Mushrooms
nb.fit5=naiveBayes(Mushrooms ~.,data=dat5)

nb_test=predict(nb.fit5, newdata=df.test[,1:10], type="raw")
RC=roc(nb_test[,2],factor(df.test$Mushrooms))
auc(RC)
nb_test=1*(nb_test[,2]>0.5)
precision(table(factor(nb_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
recall(table(factor(nb_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
1-sum((nb_test-df.test$Mushrooms)^2)/nrow(df.test)


### Removing disparate impact

## V2
cdf1=ecdf(dat$V2[dat$V3==0.48246])
cdf2=ecdf(dat$V2[dat$V3==-0.48246])
V2_f=cdf1(dat$V2[dat$V3==0.48246])
V2_m=cdf2(dat$V2[dat$V3==-0.48246])
v2_f=quantile(x=dat$V2[dat$V3==0.48246],probs=c(V2_f,V2_m))
v2_m=quantile(x=dat$V2[dat$V3==-0.48246],probs=c(V2_f,V2_m))
V2=as.numeric(rowMeans(cbind(v2_f,v2_m)))

## V4
cdf1=ecdf(dat$V4[dat$V3==0.48246])
cdf2=ecdf(dat$V4[dat$V3==-0.48246])
V4_f=cdf1(dat$V4[dat$V3==0.48246])
V4_m=cdf2(dat$V4[dat$V3==-0.48246])
v4_f=quantile(x=dat$V4[dat$V3==0.48246],probs=c(V4_f,V4_m))
v4_m=quantile(x=dat$V4[dat$V3==-0.48246],probs=c(V4_f,V4_m))
V4=as.numeric(rowMeans(cbind(v4_f,v4_m)))

## V5
cdf1=ecdf(dat$V5[dat$V3==0.48246])
cdf2=ecdf(dat$V5[dat$V3==-0.48246])
cdf_f=cdf1(dat$V5[dat$V3==0.48246])
cdf_m=cdf2(dat$V5[dat$V3==-0.48246])
v5_f=quantile(x=dat$V5[dat$V3==0.48246],probs=c(cdf_f,cdf_m))
v5_m=quantile(x=dat$V5[dat$V3==-0.48246],probs=c(cdf_f,cdf_m))
V5=as.numeric(rowMeans(cbind(v5_f,v5_m)))

## V6
cdf1=ecdf(dat$V6[dat$V3==0.48246])
cdf2=ecdf(dat$V6[dat$V3==-0.48246])
cdf_f=cdf1(dat$V6[dat$V3==0.48246])
cdf_m=cdf2(dat$V6[dat$V3==-0.48246])
v6_f=quantile(x=dat$V6[dat$V3==0.48246],probs=c(cdf_f,cdf_m))
v6_m=quantile(x=dat$V6[dat$V3==-0.48246],probs=c(cdf_f,cdf_m))
V6=as.numeric(rowMeans(cbind(v6_f,v6_m)))

## V7
cdf1=ecdf(dat$V7[dat$V3==0.48246])
cdf2=ecdf(dat$V7[dat$V3==-0.48246])
cdf_f=cdf1(dat$V7[dat$V3==0.48246])
cdf_m=cdf2(dat$V7[dat$V3==-0.48246])
v7_f=quantile(x=dat$V7[dat$V3==0.48246],probs=c(cdf_f,cdf_m))
v7_m=quantile(x=dat$V7[dat$V3==-0.48246],probs=c(cdf_f,cdf_m))
V7=as.numeric(rowMeans(cbind(v7_f,v7_m)))

## V8
cdf1=ecdf(dat$V8[dat$V3==0.48246])
cdf2=ecdf(dat$V8[dat$V3==-0.48246])
cdf_f=cdf1(dat$V8[dat$V3==0.48246])
cdf_m=cdf2(dat$V8[dat$V3==-0.48246])
v8_f=quantile(x=dat$V8[dat$V3==0.48246],probs=c(cdf_f,cdf_m))
v8_m=quantile(x=dat$V8[dat$V3==-0.48246],probs=c(cdf_f,cdf_m))
V8=as.numeric(rowMeans(cbind(v8_f,v8_m)))

## V9
cdf1=ecdf(dat$V9[dat$V3==0.48246])
cdf2=ecdf(dat$V9[dat$V3==-0.48246])
cdf_f=cdf1(dat$V9[dat$V3==0.48246])
cdf_m=cdf2(dat$V9[dat$V3==-0.48246])
v9_f=quantile(x=dat$V9[dat$V3==0.48246],probs=c(cdf_f,cdf_m))
v9_m=quantile(x=dat$V9[dat$V3==-0.48246],probs=c(cdf_f,cdf_m))
V9=as.numeric(rowMeans(cbind(v9_f,v9_m)))

## V10
cdf1=ecdf(dat$V10[dat$V3==0.48246])
cdf2=ecdf(dat$V10[dat$V3==-0.48246])
cdf_f=cdf1(dat$V10[dat$V3==0.48246])
cdf_m=cdf2(dat$V10[dat$V3==-0.48246])
v10_f=quantile(x=dat$V10[dat$V3==0.48246],probs=c(cdf_f,cdf_m))
v10_m=quantile(x=dat$V10[dat$V3==-0.48246],probs=c(cdf_f,cdf_m))
V10=as.numeric(rowMeans(cbind(v10_f,v10_m)))

## V11
cdf1=ecdf(dat$V11[dat$V3==0.48246])
cdf2=ecdf(dat$V11[dat$V3==-0.48246])
cdf_f=cdf1(dat$V11[dat$V3==0.48246])
cdf_m=cdf2(dat$V11[dat$V3==-0.48246])
v11_f=quantile(x=dat$V11[dat$V3==0.48246],probs=c(cdf_f,cdf_m))
v11_m=quantile(x=dat$V11[dat$V3==-0.48246],probs=c(cdf_f,cdf_m))
V11=as.numeric(rowMeans(cbind(v11_f,v11_m)))

## V12
cdf1=ecdf(dat$V12[dat$V3==0.48246])
cdf2=ecdf(dat$V12[dat$V3==-0.48246])
cdf_f=cdf1(dat$V12[dat$V3==0.48246])
cdf_m=cdf2(dat$V12[dat$V3==-0.48246])
v12_f=quantile(x=dat$V12[dat$V3==0.48246],probs=c(cdf_f,cdf_m))
v12_m=quantile(x=dat$V12[dat$V3==-0.48246],probs=c(cdf_f,cdf_m))
V12=as.numeric(rowMeans(cbind(v12_f,v12_m)))

## V13
cdf1=ecdf(dat$V13[dat$V3==0.48246])
cdf2=ecdf(dat$V13[dat$V3==-0.48246])
cdf_f=cdf1(dat$V13[dat$V3==0.48246])
cdf_m=cdf2(dat$V13[dat$V3==-0.48246])
v13_f=quantile(x=dat$V13[dat$V3==0.48246],probs=c(cdf_f,cdf_m))
v13_m=quantile(x=dat$V13[dat$V3==-0.48246],probs=c(cdf_f,cdf_m))
V13=as.numeric(rowMeans(cbind(v13_f,v13_m)))

## New data
dat_new=cbind(V2,dat$V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,dat[,c(13:17)])
summary(dat_new)

## Training and testing data. 70% training 30% testing
set.seed(123)
n=round(nrow(dat_new)*.7)
train = sample(1:nrow(dat_new), size=n)
df.train=dat_new[train,]
df.test=dat_new[-train,]

## Logistic regression

## For Cannabis
dat1=df.train[,1:13]
log.fit1=glm(Cannabis ~.,data=dat1,family="binomial")

log_test=predict(log.fit1, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$Cannabis))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Cannabis)), relevant='1')
1-sum((log_test-df.test$Cannabis)^2)/nrow(df.test)

## For Coke
dat2=df.train[,c(1:12,14)]
log.fit2=glm(Coke ~.,data=dat2,family="binomial")

log_test=predict(log.fit2, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$Coke))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Coke)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Coke)), relevant='1')
1-sum((log_test-df.test$Coke)^2)/nrow(df.test)

## For Ecstasy
dat3=df.train[,c(1:12,15)]
log.fit3=glm(Ecstasy ~.,data=dat3,family="binomial")

log_test=predict(log.fit3, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$Ecstasy))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Ecstasy)), relevant='1')
1-sum((log_test-df.test$Ecstasy)^2)/nrow(df.test)


## For LSD
dat4=df.train[,c(1:12,16)]
log.fit4=glm(LSD ~.,data=dat4,family="binomial")

log_test=predict(log.fit4, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$LSD))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$LSD)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$LSD)), relevant='1')
1-sum((log_test-df.test$LSD)^2)/nrow(df.test)


## For Mushrooms
dat5=df.train[,c(1:12,17)]
log.fit5=glm(Mushrooms ~.,data=dat5,family="binomial")

log_test=predict(log.fit5, newdata=df.test[,1:12], type="response")
RC=roc(log_test,factor(df.test$Mushrooms))
auc(RC)
log_test[log_test>0.5]=1
log_test[log_test<0.5]=0
precision(table(factor(log_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
recall(table(factor(log_test,c(0,1)),factor(df.test$Mushrooms)), relevant='1')
1-sum((log_test-df.test$Mushrooms)^2)/nrow(df.test)

