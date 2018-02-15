setwd("~/Github Folder/cat_ModelVariant")

data = read.csv("FilteredTestFile.csv")
str(data) # check data structure

# search Features for key words and form a new column to merge with dataframe
data$Features = as.character(data$Features)


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
# consider using data.table package
install.packages("data.table")
library(data.table)
DT = data.table(data)
summary(DT)
DT[, grep("ABS", strsplit(Seller_Comments, " " ))]


#####################

# extract relevant variables and form new dataframe

testdata = cbind.data.frame(
  data$Brand, data$Model, data$Model.variant, data$Mfg_Year, data$Engine_Capacity,
  data$Airbag, data$Leather, data$Nav, data$ABS, 
  data$Sport.Rims, data$Reverse.Camera
)

names(testdata) = c("Brand", "Model", "Modelvar", "MfgYr", "CC",
                    "Airbag", "Leather", "Nav", "ABS",
                    "SportRims", "RevCam"
                    )

levels(factor(testdata$Brand))
levels(factor(testdata$Model))
levels(testdata$Modelvar) # shows 485 model variants, Excel file shows 468 model variants

testdata$Modelvar = tolower(testdata$Modelvar) # convert all text to lower case
testdata$Modelvar = as.factor(testdata$Modelvar) # now dataframe shows 468 model variants



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
plyr::count(testdata_hon, "Modelvar")

testdata_hon = droplevels(testdata_hon) # to refactor for Honda City Model Var only
testdata_hon$MfgYr = as.numeric(testdata_hon$MfgYr) # convert yrs to int
testdata_hon[,4:5] = scale(testdata_hon[,4:5]) # scale & center "MfgYr" & "CC"

str(testdata_hon)


# NO LONGER REQUIRED: drop levels from all factor columns, alternative can use droplevels()
# testdata_hon[] <- lapply(testdata_hon, function(x) if(is.factor(x)) factor(x) else x)
# levels((testdata_hon$Modelvar)) # 11 variants for Honda City



# from the plot, most key specs cant differentiate modelvar due to sparsity of data
# mfg yr has some interesting info, models tend to aggregate in a  way
plot(testdata_hon)


# Multinomial Logistic ----------------------------------------------------

# try multinomial model, logistic model first
install.packages("nnet")
library(nnet)

multinom_hon = multinom(Modelvar ~ MfgYr, data = testdata_hon, trace=F)

summary(multinom_hon) # r.d. 3010, AIC 3050
fitted(multinom_hon)
# model.matrix(multinom_hon)

coef(multinom_hon) # baseline is model 1.3 (A)
# fitted model is log[P(Y=j|x)/P(Y=1|x)] = intercept_j + 
# \beta_{j1}*MfgYr2003 + ... + \beta_{j14}*MfgYr2016 for j=2,...,14

newdata_yr = data.frame(MfgYr=14)
predict(multinom_hon, newdata_yr)

# compare with null model
multinom_hon_null = multinom(Modelvar ~ 1, data = testdata_hon, trace=F)
anova(multinom_hon_null, multinom_hon) 
# Pr(Chi) less than 0.05, reject H0: null model is adequate

# consider adding more predictors?? key specs?? might be too sparse
# try adding CC as predictor

plot(testdata_hon$Modelvar, testdata_hon$CC)
multinom_hon1 = multinom(Modelvar ~ MfgYr + CC, data = testdata_hon, trace=F)

summary(multinom_hon1) # r.d. 3077, AIC 3137
anova(multinom_hon, multinom_hon1) # Pr(Chisq)>0.05, accept H0

multinom_hon2 = multinom(Modelvar ~ CC, data = testdata_hon, trace=F)
summary(multinom_hon2) # higher resid dev than the MfgYr model, r.d. 6402, AIC 6442

# run an AIC loop for all predictors? or do CV??
str(testdata_hon)
multinom_hon_full = multinom(Modelvar ~ ., data = testdata_hon[,-c(1,2)], trace=F) # create full model

summary(multinom_hon_full)
anova(multinom_hon, multinom_hon_full) # sig to reject H0: model with MfgYr only is suff


multinom_full_pred = predict(multinom_hon_full, newdata=testdata_hon[,-c(1,2)])
levels(multinom_full_pred); levels(testdata_hon[,3])

# compare predicted vs actual
cbind(plyr::count(multinom_full_pred), plyr::count(testdata_hon[,3]))

1-sum(multinom_full_pred==testdata_hon[,3])/nrow(testdata_hon) # 36.27% error rate, 63.7% accuracy rate

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
