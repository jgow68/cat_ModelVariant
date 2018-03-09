
setwd("~/Github Folder/cat_ModelVariant")

data = read.csv("FilteredTestFile.csv")
str(data) # check data structure

# search Features for key words and form a new column to merge with dataframe
data$Features = as.character(data$Features)


# check if R grepl function matches output with Excel search functions
testa = grepl("ABS", data$Features) # returns TRUE/FALSE for selected keyword search, add ignore.case=T to ignore case matching
sum(testa == data$ABS)/nrow(data) # checked correct

testb = grepl("Airbag", data$Features)
sum(testb == data$Airbag)/nrow(data) # checked correct

testc = grepl("Leather", data$Features)
sum(testc == data$Leather)/nrow(data) # checked correct

testd = grepl("Nav", data$Features)
sum(testd == data$Nav)/nrow(data) # checked correct

teste = grepl("Sport Rims", data$Features)
sum(teste == data$Sport.Rims)/nrow(data) # checked correct

testf = grepl("Reverse Camera", data$Features)
sum(testf == data$Reverse.Camera)/nrow(data) # checked correct



#####################
## consider using data.table package
#install.packages("data.table")
#library(data.table)
#DT = data.table(data)
#summary(DT)
#DT[, grep("ABS", strsplit(Seller_Comments, " " ))]
#####################

# extract relevant variables and form new dataframe
str(data)

testdata = cbind.data.frame(
  data$ID, data$Brand, data$Model, data$Model.Variant, data$Mfg_Year, data$Engine_Capacity, data$Transmission,
  data$Airbag, data$Leather, data$Nav, data$ABS, 
  data$Sport.Rims, data$Reverse.Camera
)

names(testdata) = c("ID", "Brand", "Model", "Modelvar", "MfgYr", "CC", "Transm",
                    "Airbag", "Leather", "Nav", "ABS",
                    "SportRims", "RevCam"
                    )

levels(testdata$Modelvar)
levels(data$Model.Variant)
all.equal(levels(testdata$Modelvar), levels(data$Model.Variant)) # check if no. of model variants matches Excel file, 468


# Test Dataset ----------------------------------------------------

# Prepare Test Dataset
testfile = read.csv("TestFile.csv")
str(testfile)

testfile_fil = cbind.data.frame(
  testfile$ID, testfile$Brand, testfile$Model, testfile$Model.Variant, testfile$Mfg_Year, testfile$Engine_Capacity, testfile$Transmission,
  testfile$Airbag, testfile$Leather, testfile$Nav, testfile$ABS, 
  testfile$Sport.Rims, testfile$Reverse.Camera
)

names(testfile_fil) = c("ID", "Brand", "Model", "Modelvar", "MfgYr", "CC", "Transm",
                    "Airbag", "Leather", "Nav", "ABS",
                    "SportRims", "RevCam"
)

str(testfile_fil)
testfile_fil$MfgYr = as.numeric(testfile_fil$MfgYr) # convert yrs to int
testfile_fil$MfgYr = as.numeric(scale(testfile_fil$MfgYr)) # scale & center numeric predictors, then coerce it back to numeric
testfile_fil$CC = as.numeric(scale(testfile_fil$CC))

test_city = dplyr::filter(testfile_fil, Brand=="Honda" & Model=="City")
plyr::count(test_city, "Modelvar") #check for fun, model var not fully completed yet
str(test_city)

test_myvi_man = dplyr::filter(testfile_fil, Brand=="Perodua" & Model=="MyVi" & Transm=="Manual")
str(test_myvi_man)

test_myvi_auto = dplyr::filter(testfile_fil, Brand=="Perodua" & Model=="MyVi" & Transm=="Auto")
str(test_myvi_auto)

test_vios_man = dplyr::filter(testfile_fil, Brand=="Toyota" & Model=="Vios" & Transm=="Manual")
str(test_vios_man)

test_vios_auto = dplyr::filter(testfile_fil, Brand=="Toyota" & Model=="Vios" & Transm=="Auto")
str(test_vios_auto)

# testdata$Modelvar = tolower(testdata$Modelvar) # convert all text to lower case
# testdata$Modelvar = as.factor(testdata$Modelvar) # now dataframe shows 468 model variants



# NO LONGER REQUIRED: Fix the duplicated levels for Honda City
# which(testdata_hon$Modelvar=="V (New model)")
# install.packages("forcats")
# library(forcats)

# testest = factor(testdata_hon$Modelvar)
# testdata_hon$Modelvar=fct_collapse(testdata_hon$Modelvar,"V (New Model)" = c("V (New model)", "V (New Model)"),
#                          "E (New Model)" = c("E (New model)", "E (New Model)")
# )






# Case: Honda City only ---------------------------------------------------

# filter data for Honda cars only

library(dplyr)
testdata_hon = dplyr::filter(testdata, Brand=="Honda" & Model=="City")
plyr::count(testdata_hon, "Modelvar") # check with Excel model variants
levels(testdata_hon$Modelvar)[levels(testdata_hon$Modelvar)%in%c("E (New model)", "E (New Model)")] = "E (New Model)"



testdata_hon = droplevels(testdata_hon) # to refactor for Honda City Model Var only
testdata_hon$MfgYr = as.numeric(testdata_hon$MfgYr) # convert yrs to int
testdata_hon$MfgYr = as.numeric(scale(testdata_hon$MfgYr)) # scale & center numeric predictors, then coerce it back to numeric
testdata_hon$CC = as.numeric(scale(testdata_hon$CC))

str(testdata_hon)

# NO LONGER REQUIRED: drop levels from all factor columns, alternative can use droplevels()
# testdata_hon[] <- lapply(testdata_hon, function(x) if(is.factor(x)) factor(x) else x)
# levels((testdata_hon$Modelvar)) # 11 variants for Honda City



# from the plot, most key specs cant differentiate modelvar due to sparsity of data
# mfg yr has some interesting info, models tend to aggregate in a  way
plot(testdata_hon)


# try multinomial model, logistic model first
install.packages("nnet")
library(nnet)

# CHECK predictors to be removed
str(testdata_hon)
city_full = multinom(Modelvar ~ MfgYr, data = testdata_hon[,-c(1,2,3,7)], trace=F)
city_pred = predict(city_full, newdata=testdata_hon[,-c(1,2,3,7)])

levels(city_pred);levels(testdata_hon$Modelvar) # 5 levels
plyr::count(city_pred); plyr::count(testdata_hon$Modelvar)

cbind(plyr::count(city_pred), plyr::count(testdata_hon$Modelvar))
1-sum(city_pred==testdata_hon$Modelvar)/nrow(testdata_hon) # rough gauge 37.78% error

city_out = cbind.data.frame(testdata_hon[,1], city_pred)
write.csv(city_out, "city_out.csv")

city_test_pred = predict(city_full, newdata=test_city[,-c(1,2,3,4,7)])
city_out = cbind.data.frame(test_city[,1], city_test_pred)


############ Additional work NOT REQUIRED

#summary(multinom_hon) # r.d. 3010, AIC 3050
#fitted(multinom_hon)
# model.matrix(multinom_hon)

#coef(multinom_hon) # baseline is model 1.3 (A)
# fitted model is log[P(Y=j|x)/P(Y=1|x)] = intercept_j + 
# \beta_{j1}*MfgYr2003 + ... + \beta_{j14}*MfgYr2016 for j=2,...,14

#newdata_yr = data.frame(MfgYr=14)
#predict(multinom_hon, newdata_yr)

# compare with null model
#multinom_hon_null = multinom(Modelvar ~ 1, data = testdata_hon, trace=F)
#anova(multinom_hon_null, multinom_hon) 
# Pr(Chi) less than 0.05, reject H0: null model is adequate

# consider adding more predictors?? key specs?? might be too sparse
# try adding CC as predictor

#plot(testdata_hon$Modelvar, testdata_hon$CC)
#multinom_hon1 = multinom(Modelvar ~ MfgYr + CC, data = testdata_hon, trace=F)

#summary(multinom_hon1) # r.d. 3077, AIC 3137
#anova(multinom_hon, multinom_hon1) # Pr(Chisq)>0.05, accept H0

#multinom_hon2 = multinom(Modelvar ~ CC, data = testdata_hon, trace=F)
#summary(multinom_hon2) # higher resid dev than the MfgYr model, r.d. 6402, AIC 6442


# CV for multinom model ---------------------------------------------------

install.packages("plyr")
library(plyr)
count(testdata_hon, "Brand") # 1817 data points, sufficient for 5-fold CV?

# try Cross Validation (5-fold)
str(testdata_hon)

CV_values = vector(length=1)
n=nrow(testdata_hon)
for(i in 1){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = multinom(testdata_hon[-k,3] ~ ., data = testdata_hon[-k, c(4:11)], trace=F) 
    yhat = predict(set_model, newdata=testdata_hon[k,c(4:11) ])
    yhat = factor(yhat, levels=levels(testdata_hon[,3])) # realign factor levels
    cvi = cvi + (1 - sum(yhat==testdata_hon[k,3])/length(yhat))
  }
  CV_values[i] = cvi/5
}

CVoutput_full = CV_values
CVoutput_full # full model has an error rate of 42.69%
levels(yhat); levels(testdata_hon[k,3])

plyr::count(yhat); plyr::count(testdata_hon[k,3])

# warning msg: group 1.3 (A) , e (new model) is empty
# reason: when doing fold CV, sparse data wont be available in some folds





# test for models with one predictor removed

CV_values = vector(length=length(x))
n=length(y)
for(i in 1:length(x)){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = multinom(y[-k,] ~ ., data = x[-k, -i], trace=F) 
    yhat = predict(set_model, newx=x[k, -i])
    cvi = cvi + (1 - sum(yhat==y[k,])/length(yhat))
  }
  CV_values[i] = cvi/5
}

CVoutput = cbind(colnames(testdata_hon[,-c(1:3)]), CV_values)


colnames(CVoutput) = c("Predictor Removed", "CV values")
CVoutput

CVoutput[which.min(CVoutput[,2]), 1]; min(CVoutput[,2]) # shows the least significant predictor, and the model's CV with that predictor removed
# 79.40% error rate, slightly better than the full model

# next step is to test for simpler models (less two predictors)
models <- lapply(paste("Modalvar", names(testdata_hon)[-3], sep = "~"), formula)
models

evars = names(testdata_hon[,-c(1:3)])
ii = lapply(1:9, seq_len)
lapply(ii, function(X)
  {(multinom(reformulate(response="Modelvar", termlabels=evars[X]), 
                data=testdata_hon[,-c(1:2)])
      )
  }
)

testdata_hon[1,]
ii = combn(seq(1:8), 7)
a = as.data.frame(0); b = as.data.frame(0)

# printed all variations of possible models based on the number of variables
for(i in 1:8){
 a = reformulate(response="Modelvar", termlabels=evars[ii[,i]])
 print(a)
}



# Case: Perodua Myvi only ---------------------------------------------------

# filter data for Myvi only

library(dplyr)
testdata_myvi = dplyr::filter(testdata, Brand=="Perodua" & Model=="MyVi")
plyr::count(testdata_myvi, "Modelvar")

str(testdata_myvi)
testdata_myvi = droplevels(testdata_myvi)
testdata_myvi$MfgYr = as.numeric(testdata_myvi$MfgYr) # convert yrs to int
testdata_myvi$MfgYr = as.numeric(scale(testdata_myvi$MfgYr)) # scale & center numeric predictors, then coerce it back to numeric
testdata_myvi$CC = as.numeric(scale(testdata_myvi$CC))

str(testdata_myvi)

# combined duplicate levels
levels(testdata_myvi$Modelvar)
levels(testdata_myvi$Modelvar)[levels(testdata_myvi$Modelvar)%in%c("1.3 (A) EZi", "1.3 (A) EZI")] = "1.3 (A) EZi"
levels(testdata_myvi$Modelvar)[levels(testdata_myvi$Modelvar)%in%c("1.3 (M) Sxi", "1.3 (M) SXi", "1.3 (M) SXI")] = "1.3 (M) SXi"

levels(testdata_myvi$Modelvar)
levels(droplevels(testdata$Modelvar[testdata$Model=="MyVi"]))
all.equal(levels(testdata_myvi$Modelvar), levels(droplevels(testdata$Modelvar[testdata$Model=="MyVi"]))) # won't be equal, manual check on Excel pivot table


############
# alternative soln to combine levels
# ha <- list(
#  unknown = c("unemployed","unknown","self-employed"),
#  class1  = c("admin.","management")
#)

#for (i in 1:length(ha)) levels(z)[levels(z)%in%ha[[i]]] <- names(ha)[i]
#############


# should split manual / auto
testdata_myvi_man = dplyr::filter(testdata_myvi, Transm=="Manual")
testdata_myvi_auto = dplyr::filter(testdata_myvi, Transm=="Auto")
plyr::count(testdata_myvi_man, "Modelvar")
plyr::count(testdata_myvi_auto, "Modelvar")

# model variants data
# 1.3 (A) EZI, 1.3 (A) EZi, 1.3 EZI (A)
# 1.3 (A) EZ, 1.3 EZ (A)
# 1.3 (A) SE, 1.3 SE (A)

# misclassify in actual data
# adj. 1.3 (m) sx and 1.3 (m) sxi to 1.3  (a) se
# 1.3 (a) EZI to 1.3 (m) sx

# remove 1.3 (a) ezi from myvi manual database
testdata_myvi_man = testdata_myvi_man[-c(which(testdata_myvi_man$Modelvar=="1.3 (A) EZ"), which(testdata_myvi_man$Modelvar=="1.3 (A) SE")), ]

# remove 1.3 (m) sx and 1.3 (m) sxi from myvi auto database
testdata_myvi_auto = testdata_myvi_auto[-c(which(testdata_myvi_auto$Modelvar=="1.3 (M) SE"), 
                                             which(testdata_myvi_auto$Modelvar=="1.3 (M) SX")), ]


# remove obs w ith frequency less than 10
testdata_myvi_auto <- testdata_myvi_auto[!(as.numeric(testdata_myvi_auto$Modelvar) %in% which(table(testdata_myvi_auto$Modelvar)<10)),]
plyr::count(testdata_myvi_auto, "Modelvar")
testdata_myvi_auto = droplevels(testdata_myvi_auto)

testdata_myvi_man <- testdata_myvi_man[!(as.numeric(testdata_myvi_man$Modelvar) %in% which(table(testdata_myvi_man$Modelvar)<10)),]
plyr::count(testdata_myvi_man, "Modelvar")
testdata_myvi_man = droplevels(testdata_myvi_man)



# build multinom model

# MyVi manual models only
# CHECK: which predictors to remove from the training data set
str(testdata_myvi_man)

myvi_man_full = multinom(Modelvar ~ ., data=testdata_myvi_man[,-c(1,2,3,7)])
myvi_man_pred = predict(myvi_man_full, newdata=testdata_myvi_man[,-c(1,2,3,7)])

levels(myvi_man_pred);levels(testdata_myvi_man$Modelvar) # 5 levels
cbind(plyr::count(myvi_man_pred), plyr::count(testdata_myvi_man$Modelvar))
1-sum(myvi_man_pred==testdata_myvi_man$Modelvar)/nrow(testdata_myvi_man) # rough gauge 34.75% error

# MyVi auto models only
myvi_auto_full = multinom(Modelvar ~ ., data=testdata_myvi_auto[,-c(1,2,3,7)])
myvi_auto_pred = predict(myvi_auto_full, newdata=testdata_myvi_auto[,-c(1,2,3,7)])
levels(myvi_auto_pred);levels(testdata_myvi_auto$Modelvar) # 7 levels
cbind(plyr::count(myvi_auto_pred), plyr::count(testdata_myvi_auto$Modelvar)) # failed
plyr::count(myvi_auto_pred); plyr::count(testdata_myvi_auto$Modelvar)
1-sum(myvi_auto_pred==testdata_myvi_auto$Modelvar)/nrow(testdata_myvi_auto) # rough gauge 37.93% err

# output predictions
# test on our training set first
# link output to data obs ID
myvi_man_output = cbind.data.frame(testdata_myvi_man[,1], myvi_man_pred)
myvi_auto_output = cbind.data.frame(testdata_myvi_auto[,1], myvi_auto_pred)

names(myvi_man_output) = c("ID", "EstModelVar")
names(myvi_auto_output) = c("ID", "EstModelVar")


myvi_output = rbind.data.frame(myvi_man_output, myvi_auto_output)
write.csv(myvi_output, "myvi_out.csv")


# predict based on test dataset

myvi_man_test_pred = predict(myvi_man_full, newdata=test_myvi_man[,-c(1,2,3,4,7)])
myvi_auto_test_pred = predict(myvi_auto_full, newdata=test_myvi_auto[,-c(1,2,3,4,7)])

myvi_man_test_output = cbind.data.frame(test_myvi_man[,1], myvi_man_test_pred)
myvi_auto_test_output = cbind.data.frame(test_myvi_auto[,1], myvi_auto_test_pred)

names(myvi_man_test_output) = c("ID", "EstModelVar")
names(myvi_auto_test_output) = c("ID", "EstModelVar")

myvi_test_output = rbind.data.frame(myvi_man_test_output, myvi_auto_test_output)
write.csv(myvi_test_output, "myvi_out.csv")

# Case: Toyota Vios -------------------------------------------------------------

testdata_vios =  dplyr::filter(testdata, Brand=="Toyota" & Model=="Vios")
plyr::count(testdata_vios, "Modelvar")

str(testdata_vios)
testdata_vios = droplevels(testdata_vios)
testdata_vios$MfgYr = as.numeric(testdata_vios$MfgYr) # convert yrs to int
testdata_vios$MfgYr = as.numeric(scale(testdata_vios$MfgYr)) # scale & center numeric predictors, then coerce it back to numeric
testdata_vios$CC = as.numeric(scale(testdata_vios$CC))


# split Auto/Manual
testdata_vios_man = dplyr::filter(testdata_vios, Transm=="Manual")
testdata_vios_auto = dplyr::filter(testdata_vios, Transm=="Auto")

testdata_vios_man = droplevels(testdata_vios_man)
testdata_vios_auto = droplevels(testdata_vios_auto)

plyr::count(testdata_vios_man, "Modelvar")
plyr::count(testdata_vios_auto, "Modelvar")

# remove obs that is misclassify
testdata_vios_man = testdata_vios_man[-c(which(testdata_vios_man$Modelvar=="J 1.5 (A)"), which(testdata_vios_man$Modelvar=="S 1.5 (A)")), ]

# Vios Manual
vios_man = multinom(Modelvar ~ ., data=testdata_vios_man[,-c(1,2,3,7)])
vios_man_pred = predict(vios_man, newdata=testdata_vios_man[,-c(1,2,3,7)])
levels(vios_man_pred);levels(testdata_vios_man$Modelvar) # 3 levels
cbind(plyr::count(vios_man_pred), plyr::count(testdata_vios_man$Modelvar)) # failed
plyr::count(vios_man_pred); plyr::count(testdata_vios_man$Modelvar)
1-sum(vios_man_pred==testdata_vios_man$Modelvar)/nrow(testdata_vios_man) # rough gauge 7.3% err

# Vios Auto
vios_auto = multinom(Modelvar ~ ., data=testdata_vios_auto[,-c(1,2,3,7)])
vios_auto_pred = predict(vios_auto, newdata=testdata_vios_auto[,-c(1,2,3,7)])
levels(vios_auto_pred);levels(testdata_vios_auto$Modelvar) # 13 levels
cbind(plyr::count(vios_auto_pred), plyr::count(testdata_vios_auto$Modelvar)) # failed
plyr::count(vios_auto_pred); plyr::count(testdata_vios_auto$Modelvar)
1-sum(vios_auto_pred==testdata_vios_auto$Modelvar)/nrow(testdata_vios_auto) # rough gauge 57.2% err

# test on test dataset

vios_auto_test_pred = predict(vios_auto, newdata=test_vios_auto[,-c(1,2,3,4,7)])
vios_man_test_pred = predict(vios_auto, newdata=test_vios_man[,-c(1,2,3,4,7)])

vios_auto_out = cbind.data.frame(test_vios_auto[,1], vios_auto_test_pred)
vios_man_out = cbind.data.frame(test_vios_man[,1], vios_man_test_pred)

names(vios_auto_out) = c("ID", "EstModelVar")
names(vios_man_out) = c("ID", "EstModelVar")

vios_test_out = rbind.data.frame(vios_auto_out, vios_man_out)


# Consolidate all outputs into one file -------------------------------------------------

# align col names
names(myvi_test_output) = c("ID", "EstModelVar")
names(city_out) = c("ID", "EstModelVar")
names(vios_test_out) = c("ID", "EstModelVar")


pred_all = rbind.data.frame(city_out, myvi_test_output, vios_test_out)
write.csv(pred_all, "pred_all.csv")


# Caret - multinomial -----------------------------------------------------

# caret




# glmnet multinomial ------------------------------------------------------
install.packages("glmnet")
library(glmnet)

x = as.matrix(x)
y = as.matrix(y)
str(x); str(y)
summary(x,y)

x1 = model.matrix(~.-1, x)
y1 = model.matrix(~. -1, y)
str(x1); str(y1);

# issues in lambda convergence ??
fit = glmnet(x1, y1, family="multinomial")
plot(fit)

predict(fit, newx=x1[1,])

cvfit=cv.glmnet(x1, y1, family="multinomial")
plot(cvfit)











# try AIC, stepAIC issues, not reliable anyway
install.packages("MASS")
library(MASS)
stepAIC(multinom_hon_null, direction="both",trace=TRUE) # issues, cant step up?


# SVM approach ------------------------------------------------------------
install.packages("e1071")
library(e1071)

svm_test = svm(Modelvar ~ ., data = testdata_hon[,-c(1,2)])
svm_pred = predict(svm_test, newdata=testdata_hon[,-c(1,2)])
1-sum(testdata_hon[,3]==svm_pred)/nrow(testdata_hon) # 38.47% error rate, 61.53% accuracy rate

plyr::count(svm_pred); plyr::count(testdata_hon, vars="Modelvar")
# model can't predict any of the "S" variants, mostly categorized into "E"


# SVM CV ------------------------------------------------------------------


# CV the manual way
svm_test = svm(Modelvar ~ ., data = testdata_hon[-c(1:50),-c(1,2)]) # train on the data excl. first 50
svm_pred = predict(svm_test, newdata=testdata_hon[c(1:50),-c(1,2)]) # predict based on first 50 predictors
sum(testdata_hon[c(1:50),3]==svm_pred)/nrow(testdata_hon[c(1:50),]) # 82% accuracy rate

x2 = as.data.frame(testdata_hon[,-c(1,2)])
y2 = as.data.frame(testdata_hon[,3])

CV_values_svm = vector(length=1)
n=length(testdata_hon[,3])
for(i in 1){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = svm(testdata_hon[-k,3] ~ ., data = testdata_hon[-k, -c(1,2)], trace=F) 
    yhat = predict(set_model, newdata=testdata_hon[k, -c(1,2)])
    cvi = cvi + (1 - sum(yhat==testdata_hon[k,3])/length(yhat))
  }
  CV_values_svm[i] = cvi/5
}

CV_values_svm

levels(yhat); levels(testdata_hon[,3])
plyr::count(yhat); plyr::count(testdata_hon[k,3])
# at the last fold CV, as expected, due to lack of data of the new car models,
# our svm model prediction led to more basic models "E" & "S"
# the model has a higher success rate for predicting basic models

# consider doing 5-fold CV to compare with multinom model
install.packages("caret")
library(caret)

# rename model names due to caret req
head(testdata_hon)
testdata_hon[,3] = as.factor(make.names(testdata_hon[,3]))
str(testdata_hon[,3])

ctrl=trainControl(method="cv", savePred=T, classProbs=T)
svm_caret = train(Modelvar ~ ., data = testdata_hon[,-c(1,2)], 
                  method="svmRadialWeights", trControl=ctrl)
head(svm_caret$pred)


# Other Methods (Issues) --------------------------------------------------

# try k-means, doesnt work, requrire data to be numeric matrix
# consider converting factors to numbers?
set.seed(134)
hon_output = kmeans(testdata_hon, centers=1)


# try rpart - not useful, meant more for binary trees
install.packages("rpart")
library(rpart)
hon_output = rpart(Modelvar ~ Airbag + Leather + Nav +
                 ABS + SportRims + RevCam,
                 method="class",
      data = testdata_hon
      )

printcp(hon_output)
plot(hon_output)
plotcp(hon_output)
