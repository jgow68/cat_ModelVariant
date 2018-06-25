

# Read data from csv ------------------------------------------------------


setwd("~/Github Folder/cat_ModelVariant")

# form training df
data1 = read.csv("Testset_Feb18_V1.csv")
data2 = read.csv('Trainingset_Jan18_V1.csv')
data3 = read.csv('Dec17_V1.csv')
data4 = read.csv('Nov17_V3.csv')

# form testing df
verify_data = read.csv("Testset_Mar18_V1.csv")
str(verify_data)

with(verify_data, mean(Price,na.rm = TRUE))

summary(verify_data$Price)

## ## ## ## ## ## ## ##  
# Not Required Now: Use Excel to add columns
## ## ## ## ## ## ## ## 

# search Features for key words and form a new column to merge with dataframe
data3$Features = as.character(data3$Features)
data4$Features = as.character(data4$Features)

data3$Airbag = grepl("Airbag", data3$Features)
data3$Leather = grepl("Leather", data3$Features)
data3$Nav = grepl("Nav", data3$Features)
data3$ABS = grepl("ABS", data3$Features)
data3$Sport.Rim = grepl("Sport rim", data3$Features) # no features extracted, consider remove
data3$Reverse.Camera = grepl("Reverse camera", data3$Features) # no features extracted, consider remove
data3$Power.Door = grepl("Power Door", data3$Features)
data3$Touchscreen = grepl("Touchscreen", data3$Features) # no features extracted, consider remove
data3$Climate.Control = grepl("Climate control", data3$Features) # no features extracted, consider remove
str(data3)
dim(data3)

data4$Airbag = grepl("Airbag", data4$Features)
data4$Leather = grepl("Leather", data4$Features)
data4$Nav = grepl("Nav", data4$Features)
data4$ABS = grepl("ABS", data4$Features)
data4$Sport.Rim = grepl("Sport rim", data4$Features) # no features extracted, consider remove
data4$Reverse.Camera = grepl("Reverse camera", data4$Features) # no features extracted, consider remove
data4$Power.Door = grepl("Power Door", data4$Features)
data4$Touchscreen = grepl("Touchscreen", data4$Features) # no features extracted, consider remove
data4$Climate.Control = grepl("Climate control", data4$Features) # no features extracted, consider remove

## ## ## ## ## ## ## ## 

## ## ## ## ## ## ## ## 

## consider using data.table package
#install.packages("data.table")
#library(data.table)
#DT = data.table(data)
#summary(DT)
#DT[, grep("ABS", strsplit(Seller_Comments, " " ))]

## ## ## ## ## ## ## ## 

# Extract data, form dataframe --------------------------------------------

# extract relevant variables and form new dataframe

training_data1 = cbind.data.frame(
  data1$ID, data1$Brand, data1$Model, data1$Features, data1$Price,
  data1$Model.Variant, data1$Mfg_Year, data1$Engine_Capacity, data1$Transmission,
  data1$Airbag, data1$Leather, data1$Nav, data1$ABS, 
  data1$Sport.Rims, data1$Reverse.Camera, data1$Power.Door,
  data1$Touchscreen, data1$Climate.Control
)

names(training_data1) = c("ID", "Brand", "Model", "Features", "Price",
                         "Modelvar", "MfgYr", "CC", "Transm",
                         "Airbag", "Leather", "Nav", "ABS",
                         "SportRims", "RevCam", "PowDoor",
                         "TouchScreen", "ClimaCtrl"
                    )

training_data2 = cbind.data.frame(
  data2$ID, data2$Brand, data2$Model, data2$Features, data2$Price,
  data2$Model.Variant, data2$Mfg_Year, data2$Engine_Capacity, data2$Transmission,
  data2$Airbag, data2$Leather, data2$Nav, data2$ABS, 
  data2$Sport.Rims, data2$Reverse.Camera, data2$Power.Door,
  data2$Touchscreen, data2$Climate.Control
)

names(training_data2) = c("ID", "Brand", "Model", "Features", "Price",
                          "Modelvar", "MfgYr", "CC", "Transm",
                          "Airbag", "Leather", "Nav", "ABS",
                          "SportRims", "RevCam", "PowDoor",
                          "TouchScreen", "ClimaCtrl"
)

training_data3 = cbind.data.frame(
  data3$ID, data3$Brand, data3$Model, data3$Features, data3$Price,
  data3$Model.variant, data3$Mfg_Year, data3$Engine_Capacity, data3$Transmission,
  data3$Airbag, data3$Leather, data3$Nav, data3$ABS, 
  data3$Sport.Rim, data3$Reverse.Camera, data3$Power.Door,
  data3$Touchscreen, data3$Climate.Control
)

names(training_data3) = c("ID", "Brand", "Model", "Features", "Price",
                          "Modelvar", "MfgYr", "CC", "Transm",
                          "Airbag", "Leather", "Nav", "ABS",
                          "SportRims", "RevCam", "PowDoor",
                          "TouchScreen", "ClimaCtrl"
)

training_data4 = cbind.data.frame(
  data4$ID, data4$Brand, data4$Model, data4$Features, data4$Price,
  data4$Model.variant, data4$Mfg_Year, data4$Engine_Capacity, data4$Transmission,
  data4$Airbag, data4$Leather, data4$Nav, data4$ABS, 
  data4$Sport.Rim, data4$Reverse.Camera, data4$Power.Door,
  data4$Touchscreen, data4$Climate.Control
)

names(training_data4) = c("ID", "Brand", "Model", "Features", "Price",
                          "Modelvar", "MfgYr", "CC", "Transm",
                          "Airbag", "Leather", "Nav", "ABS",
                          "SportRims", "RevCam", "PowDoor",
                          "TouchScreen", "ClimaCtrl"
)

str(training_data1) #56,458 obs
str(training_data2) #58,536 obs
str(training_data3) #109,892 obs
str(training_data4) #109,591 obs

# 1st case: choose most recent data set for training
training_data = training_data1

# 2nd case: merge historical datasets by outer join 2 df
# consider using data.table for faster computation
# Possible issues: join witihout specifying ID
training_data = merge(training_data1, training_data2, all=TRUE)
training_data = merge(training_data, training_data3, all=TRUE)
training_data = merge(training_data, training_data4, all=TRUE)

str(training_data) # 246,607 obs

# all.equal(levels(testdata$Modelvar), levels(data$Model.Variant)) 

testing_data = cbind.data.frame(
  verify_data$ID, verify_data$Brand, verify_data$Model, verify_data$Features, verify_data$Price,
  verify_data$Model.Variant, verify_data$Mfg_Year, verify_data$Engine_Capacity, verify_data$Transmission,
  verify_data$Airbag, verify_data$Leather, verify_data$Nav, verify_data$ABS, 
  verify_data$Sport.Rims, verify_data$Reverse.Camera, verify_data$Power.Door,
  verify_data$Touchscreen, verify_data$Climate.Control
)

names(testing_data) = c("ID", "Brand", "Model", "Features", "Price",
                         "Modelvar", "MfgYr", "CC", "Transm",
                         "Airbag", "Leather", "Nav", "ABS",
                         "SportRims", "RevCam", "PowDoor",
                         "TouchScreen", "ClimaCtrl"
)

str(testing_data)

all.equal(levels(testing_data$Modelvar), levels(training_data$Modelvar))


# Prep Training Dataset ----------------------------------------------------

str(training_data)
training_data$MfgYr = as.numeric(training_data$MfgYr) # convert "MfgYr" to int
training_data$Features = as.character(training_data$Features) # convert "Features" to text


library("scales")
training_data$MfgYr = rescale(training_data$MfgYr, to=c(0,1))
training_data$CC = rescale(training_data$CC, to=c(0,1))

# # # # # # # #
# Do by Make  #
#             #
#             #
# # # # # # # #


training_city = dplyr::filter(training_data, Brand=="Honda" & Model=="City" & Modelvar != "")
plyr::count(training_city, "Modelvar")
str(training_city)

training_myvi_man = dplyr::filter(training_data, Brand=="Perodua" & Model=="MyVi" & Transm=="Manual")
str(training_myvi_man)

training_myvi_auto = dplyr::filter(training_data, Brand=="Perodua" & Model=="MyVi" & Transm=="Auto")
str(training_myvi_auto)

training_vios_man = dplyr::filter(training_data, Brand=="Toyota" & Model=="Vios" & Transm=="Manual")
str(training_vios_man)

training_vios_auto = dplyr::filter(training_data, Brand=="Toyota" & Model=="Vios" & Transm=="Auto")
str(training_vios_auto)

# Prep Testing Dataset ----------------------------------------------------

testing_data$MfgYr = as.numeric(testing_data$MfgYr) # convert "MfgYr" to int
testing_data$CC = as.numeric(levels(testing_data$CC))[testing_data$CC]
testing_data$Features = as.character(testing_data$Features) # convert "Features" to text

testing_data$MfgYr = rescale(testing_data$MfgYr, to=c(0,1))
testing_data$CC = rescale(testing_data$CC, to=c(0,1))

testing_city = dplyr::filter(testing_data, Brand=="Honda" & Model=="City" & Modelvar != "")
str(testing_city)

testing_myvi_man = dplyr::filter(testing_data, Brand=="Perodua" & Model=="MyVi" & 
                                   Transm=="Manual" & !Modelvar %in% c("", "-"))
str(testing_myvi_man)

testing_myvi_auto = dplyr::filter(testing_data, Brand=="Perodua" & Model=="MyVi" & Transm=="Auto" & Modelvar != "")
str(testing_myvi_auto)

testing_vios_man = dplyr::filter(testing_data, Brand=="Toyota" & Model=="Vios" & Transm=="Manual" & Modelvar != "")
str(testing_vios_man)

testing_vios_auto = dplyr::filter(testing_data, Brand=="Toyota" & Model=="Vios" & Transm=="Auto" & Modelvar != "")
str(testing_vios_auto)

# NO LONGER REQUIRED: Fix the duplicated levels for Honda Citoy
# which(testdata_hon$Modelvar=="V (New model)")
# install.packages("forcats")
# library(forcats)

# testest = factor(testdata_hon$Modelvar)
# testdata_hon$Modelvar=fct_collapse(testdata_hon$Modelvar,"V (New Model)" = c("V (New model)", "V (New Model)"),
#                          "E (New Model)" = c("E (New model)", "E (New Model)")
# )


# Case: Myvi Auto --------------------------------------------------------------

plyr::count(training_myvi_auto, "Modelvar")
plyr::count(testing_myvi_auto, "Modelvar")

# convert variants to text
training_myvi_auto$Modelvar = as.character(training_myvi_auto$Modelvar)
testing_myvi_auto$Modelvar = as.character(testing_myvi_auto$Modelvar)


# clean up spaces / upper-lower char and re-factor
training_myvi_auto$Modelvar = as.factor(toupper(trimws(training_myvi_auto$Modelvar)))
testing_myvi_auto$Modelvar = as.factor(toupper(trimws(testing_myvi_auto$Modelvar)))

# occurences of manual / other variants
levels(testing_myvi_auto$Modelvar)
select_myvi_auto = c('1.3', '1.3 (A) EZ', '1.3 (A) EZI', '1.3 (A) SE', '1.3 (A) X', '1.5 SE (A)')

training_myvi_auto = training_myvi_auto[training_myvi_auto$Modelvar %in% select_myvi_auto,]
testing_myvi_auto = testing_myvi_auto[testing_myvi_auto$Modelvar %in% select_myvi_auto,]

training_myvi_auto = droplevels(training_myvi_auto)
testing_myvi_auto = droplevels(testing_myvi_auto)

myvi_auto_levels = unique(c(levels(testing_myvi_auto$Modelvar),levels(training_myvi_auto$Modelvar)))

training_myvi_auto$Modelvar = factor(training_myvi_auto$Modelvar, levels = myvi_auto_levels)
testing_myvi_auto$Modelvar = factor(testing_myvi_auto$Modelvar, levels = myvi_auto_levels)

# Case: Myvi Manual -------------------------------------------------------

training_myvi_man$Modelvar = as.character(training_myvi_man$Modelvar)
testing_myvi_man$Modelvar = as.character(testing_myvi_man$Modelvar)

training_myvi_man$Modelvar = as.factor(toupper(trimws(training_myvi_man$Modelvar)))
testing_myvi_man$Modelvar = as.factor(toupper(trimws(testing_myvi_man$Modelvar)))

plyr::count(test$Modelvar)

plyr::count(training_myvi_man, "Modelvar")
plyr::count(testing_myvi_man, "Modelvar")

# filter for variants with size > 50
training_myvi_man = training_myvi_man %>% 
  group_by(Modelvar) %>%
  filter(n()>50) %>%
  as.data.frame()

testing_myvi_man = testing_myvi_man %>% 
  group_by(Modelvar) %>%
  filter(n()>50) %>%
  as.data.frame()

training_myvi_man = droplevels(training_myvi_man)
testing_myvi_man = droplevels(testing_myvi_man)

dplyr::all_equal(levels(training_myvi_man$Modelvar), levels(testing_myvi_man$Modelvar))

# if not all_equal = False, then:
# myvi_man_levels = unique(c(levels(testing_myvi_man$Modelvar),levels(training_myvi_man$Modelvar)))
# training_myvi_man$Modelvar = factor(training_myvi_man$Modelvar, levels = myvi_man_levels)
# testing_myvi_man$Modelvar = factor(testing_myvi_man$Modelvar, levels = myvi_man_levels)



# Features Selection (Myvi Auto) -----------------------------------------------

# Identify useful predictors
plyr::count(training_myvi_auto, "Transm") # filtered for autos
plyr::count(training_myvi_auto, "CC")
plot(training_myvi_auto$CC) # 2 distinct different CCs but there are also spread of other values
plyr::count(training_myvi_auto, "Airbag")
plyr::count(training_myvi_auto, "Leather")
plyr::count(training_myvi_auto, "Nav")
plyr::count(training_myvi_auto, "ABS")
plyr::count(training_myvi_auto, "SportRims")
plyr::count(training_myvi_auto, "RevCam")
plyr::count(training_myvi_auto, "PowDoor")
plyr::count(training_myvi_auto, "TouchScreen") # all N/A
plyr::count(training_myvi_auto, "ClimaCtrl")

# drop non-useful predictors
drops = c("Transm", "TouchScreen")
training_myvi_auto = training_myvi_auto[, !names(training_myvi_auto) %in% drops]
str(training_myvi_auto)

testing_myvi_auto = testing_myvi_auto[, !names(testing_myvi_auto) %in% drops]
str(testing_myvi_auto)

test_search = grepl("leather", testing_myvi_auto$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

test_search = grepl("light", testing_myvi_auto$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

leather_train_myvi_auto = grepl("leather", training_myvi_auto$Features, ignore.case=TRUE)
leather_test_myvi_auto = grepl("leather", testing_myvi_auto$Features, ignore.case=TRUE)

light_train_myvi_auto = grepl("light", training_myvi_auto$Features, ignore.case=TRUE)
light_test_myvi_auto = grepl("light", testing_myvi_auto$Features, ignore.case=TRUE)

training_myvi_auto$Leather = leather_train_myvi_auto
testing_myvi_auto$Leather = leather_test_myvi_auto

training_myvi_auto$Light = light_train_myvi_auto
testing_myvi_auto$Light = light_test_myvi_auto

colnames(training_myvi_auto) == colnames(testing_myvi_auto)


# Feature Selection (Myvi Manual) -----------------------------------------

# Identify useful predictors
plyr::count(training_myvi_man, "Transm") # filtered for manual
plyr::count(training_myvi_man, "CC")
plot(training_myvi_man$CC) # 2 distinct different CCs but there are also spread of other values
plyr::count(training_myvi_man, "Airbag")
plyr::count(training_myvi_man, "Leather")
plyr::count(training_myvi_man, "Nav")
plyr::count(training_myvi_man, "ABS")
plyr::count(training_myvi_man, "SportRims")
plyr::count(training_myvi_man, "RevCam")
plyr::count(training_myvi_man, "PowDoor")
plyr::count(training_myvi_man, "TouchScreen") # all N/A
plyr::count(training_myvi_man, "ClimaCtrl")

# drop non-useful predictors
drops = c("Transm", "TouchScreen")
training_myvi_man = training_myvi_man[, !names(training_myvi_man) %in% drops]
# str(training_myvi_man)

testing_myvi_man = testing_myvi_man[, !names(testing_myvi_man) %in% drops]
# str(testing_myvi_man)

test_search = grepl("leather", testing_myvi_man$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

test_search = grepl("light", testing_myvi_man$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

leather_train_myvi_auto = grepl("leather", training_myvi_man$Features, ignore.case=TRUE)
leather_test_myvi_auto = grepl("leather", testing_myvi_man$Features, ignore.case=TRUE)

light_train_myvi_auto = grepl("light", training_myvi_man$Features, ignore.case=TRUE)
light_test_myvi_auto = grepl("light", testing_myvi_man$Features, ignore.case=TRUE)

training_myvi_man$Leather = leather_train_myvi_auto
testing_myvi_man$Leather = leather_test_myvi_auto

training_myvi_man$Light = light_train_myvi_auto
testing_myvi_man$Light = light_test_myvi_auto

colnames(training_myvi_man) == colnames(testing_myvi_man)

# SVM: Myvi Auto ---------------------------------------------------------

library(e1071)

# for merged dataset: took too long to run & produced error, filter first 3000 obs to train
str(training_myvi_auto)

system.time(svm_tune_radial <- tune(svm, Modelvar ~ ., data = training_myvi_auto[,-c(1:4, 8)],
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
            )
# for one dataset ~3,000 obs  # cost = 10, gamma = 0.1
# user  system elapsed 
# 341.22    1.87  345.62 

# for 6,000 obs (excl price)
# user  system elapsed 
# 1272.97    6.91 1291.64 

# for last 3,000+ obs (excl price)  # cost = 10, gamma = 1.1
# user  system elapsed 
# 539.41    3.09  568.40 

# for last 3,000+ obs (excl CC)  # cost = 100, gamma = 0.1
# user  system elapsed 
# 505.91    1.51  525.30 

svm_tune_radial$best.parameters

# error rate is still high ~50%
svm_myvi_auto = svm(Modelvar ~ ., data = training_myvi_auto[,-c(1:4, 8)], trace=F,
                    cost = svm_tune_radial$best.parameters$cost, gamma = svm_tune_radial$best.parameters$gamma)
svm_pred_myvi_auto = predict(svm_myvi_auto, newdata=testing_myvi_auto[,-c(1:4, 8)])

1-sum(testing_myvi_auto[,6]==svm_pred_myvi_auto)/nrow(testing_myvi_auto)
# Error rate
# without price predictor 46% error rate
# with price predictor 54% error rate
# w/o cc 30%


# Issues:
# svm cant predict 1.3, 1.3
# unless use more recent data

table('Model: SVM' = svm_pred_myvi_auto, testing_myvi_auto[,6])

model_price_myvi_auto = cbind.data.frame(testing_myvi_auto$Modelvar, svm_pred_myvi_auto, testing_myvi_auto$Price)
names(model_price_myvi_auto) = c('ActualModel', 'EstModel', 'Price')

myvi_auto_actual_price = model_price_myvi_auto %>% 
  group_by(ActualModel) %>%
  dplyr::summarise(Price = mean(Price))

myvi_auto_est_price = model_price_myvi_auto %>%
  group_by(EstModel) %>%
  dplyr::summarise(Price = mean(Price))

cbind.data.frame(myvi_auto_actual_price, 'Est Price' = myvi_auto_est_price$Price,
                 'Diff' = abs(myvi_auto_actual_price$Price - myvi_auto_est_price$Price))




# # # Improvements: # # #

# categorize CC / scale after filtering for models
# test caret
# test gradient boosting


# # # # # # # # # # # # #


# SVM: Myvi Manual --------------------------------------------------------

dim(training_myvi_man)
str(training_myvi_man)

system.time(svm_tune_radial <- tune(svm, Modelvar ~ ., data = training_myvi_man[,-c(1:4, 8)],
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 800 obs, (17 - x) cols
# user  system elapsed 
# 41.33    0.16   42.62 

svm_tune_radial$best.parameters 
# cost= 10, gamma=0.1 full, excl price
# cost=100, gamma=0.1 excl c(price,CC), CC

svm_myvi_man = svm(Modelvar ~ ., data = training_myvi_man[,-c(1:4, 8)], trace=F,
                    cost = 100, gamma = 0.1)
svm_pred_myvi_man = predict(svm_myvi_man, newdata=testing_myvi_man[,-c(1:4, 8)])

1-sum(testing_myvi_man[,6]==svm_pred_myvi_man)/nrow(testing_myvi_man)

# Error rate info:
# 81% full model
# 87% excl price
# 43% excl price, CC
# 66% excl CC

table('Model: SVM' = svm_pred_myvi_man, testing_myvi_man[,6]) # compare differences

model_price_myvi_man = cbind.data.frame(testing_myvi_man$Modelvar, svm_pred_myvi_man, testing_myvi_man$Price)
names(model_price_myvi_man) = c('ActualModel', 'EstModel', 'Price')

myvi_man_actual_price = model_price_myvi_man %>% 
  group_by(ActualModel) %>%
  dplyr::summarise(Price = mean(Price))

myvi_man_est_price = model_price_myvi_man %>%
  group_by(EstModel) %>%
  dplyr::summarise(Price = mean(Price))

cbind.data.frame(myvi_man_actual_price, 'Est Price' = myvi_man_est_price$Price,
                 'Diff' = abs(myvi_man_actual_price$Price - myvi_man_est_price$Price))
                   
# Case: Honda City only ---------------------------------------------------

plyr::count(training_city, "Modelvar") # check with Excel model variants
plyr::count(testing_city, "Modelvar")

training_city$Modelvar = as.character(training_city$Modelvar)
testing_city$Modelvar = as.character(testing_city$Modelvar)

training_city$Modelvar = trimws(training_city$Modelvar)
testing_city$Modelvar = trimws(testing_city$Modelvar)
# Alternative function for trim spaces: trim <- function (x) gsub("^\\s+|\\s+$", "", x)

training_city$Modelvar = as.factor(training_city$Modelvar)
testing_city$Modelvar = as.factor(testing_city$Modelvar)

# consider trim spaces first
# trimws(x, which = c("both", "left", "right"))

testing_city$Modelvar[testing_city$Modelvar %in% c("E ", "E  ")] = "E"
testing_city$Modelvar[testing_city$Modelvar %in% c("S ", "S  ")] = "S"
testing_city$Modelvar[testing_city$Modelvar %in% c("VTEC ", "VTEC  ")] = "VTEC"

testing_city$Modelvar = as.factor(testing_city$Modelvar)

str(testing_city)
levels(testing_city$Modelvar)
# merge duplicated factor classes due to Upper/Lower cases
levels(training_city$Modelvar)[levels(training_city$Modelvar)%in%c("E (New model)", "E (New Model)")] = "E (New Model)"

training_city = droplevels(training_city)
levels(training_city$Modelvar) # check if match Excel model var

levels(testing_city$Modelvar)[levels(testing_city$Modelvar)%in%c("E (New model)", "E (New Model)")] = "E (New Model)"
testing_city = droplevels(testing_city)
levels(testing_city$Modelvar)

city_levels = unique(c(levels(testing_city$Modelvar),levels(training_city$Modelvar)))
city_levels

training_city$Modelvar = factor(training_city$Modelvar, levels = city_levels)
testing_city$Modelvar = factor(testing_city$Modelvar, levels = city_levels)


# Feature selection (City) -------------------------------------------------------

# Identify useful predictors
plyr::count(training_city, "Transm") # nearly all autos
plyr::count(training_city, "CC") # all approx the same
plyr::count(training_city, "Airbag")
plyr::count(training_city, "Leather")
plyr::count(training_city, "Nav")
plyr::count(training_city, "ABS")
plyr::count(training_city, "SportRims")
plyr::count(training_city, "RevCam")
plyr::count(training_city, "PowDoor")
plyr::count(training_city, "TouchScreen") # all N/A
plyr::count(training_city, "ClimaCtrl")

# drop non-useful predictors
drops = c("Transm", "CC", "TouchScreen")
training_city = training_city[, !names(training_city) %in% drops]
str(training_city)

testing_city = testing_city[, !names(testing_city) %in% drops]
str(testing_city)

padshift_train_city = grepl("paddle", training_city$Features, ignore.case=TRUE)
plyr::count(padshift_train_city)

training_city$PadShift = padshift_train_city

padshift_test_city = grepl("paddle", testing_city$Features, ignore.case=TRUE)
plyr::count(padshift_test_city)

testing_city$PadShift = padshift_test_city


# Alternative keep useful predictor method
# keeps = c("a", "b")
# df[keeps, drop=FALSE]

# NO LONGER REQUIRED: drop levels from all factor columns, alternative can use droplevels()
# testdata_hon[] <- lapply(testdata_hon, function(x) if(is.factor(x)) factor(x) else x)
# levels((testdata_hon$Modelvar))

# Mulitinomial model ------------------------------------------------------

library(nnet)

multinom_city = multinom(Modelvar ~ ., data = training_city[,-c(1:4)], trace=F)
multinom_city_pred = predict(multinom_city, newdata=testing_city[,-c(1:4)])

all.equal(levels(testing_city$Modelvar),levels(multinom_city_pred))
levels(testing_city$Modelvar)
levels(multinom_city_pred)
1-sum(multinom_city_pred==testing_city$Modelvar)/length(multinom_city_pred) 
# 33.63% error (Feb'18 test Mar'18)
# 33.31% error (Jan'18 test Feb'18)
# 35.31% error (merged Jan-Feb'18 test Mar'18)
table(multinom_city_pred, testing_city$Modelvar)

city_out = cbind.data.frame(testdata_hon[,1], city_pred)
write.csv(city_out, "city_out.csv")

city_test_pred = predict(city_full, newdata=test_city[,-c(1,2,3,4,7)])
city_out = cbind.data.frame(test_city[,1], city_test_pred)



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



# SVM approach ------------------------------------------------------------

library(e1071)

head(training_city)
svm_test = svm(Modelvar ~ ., data = training_city[,-c(1:4)])
svm_pred = predict(svm_test, newdata=training_city[,-c(1:4)])
1-sum(training_city[,5]==svm_pred)/nrow(training_city) # 32.03% error rate

plyr::count(svm_pred); plyr::count(training_city, vars="Modelvar")
# model can't predict any of the "S" variants, est. categorized into "E"

# SVM CV (skip)------------------------------------------------------------------


# CV the manual way
svm_test = svm(Modelvar ~ ., data = testdata_hon[-c(1:50),-c(1:4)]) # train on the data excl. first 50
svm_pred = predict(svm_test, newdata=testdata_hon[c(1:50),-c(1:4)]) # predict based on first 50 predictors
sum(testdata_hon[c(1:50),5]==svm_pred)/nrow(testdata_hon[c(1:50),]) # 82% accuracy rate

x2 = as.data.frame(training_city[,-c(1:4)])
y2 = as.data.frame(training_city[,5])

CV_values_svm = vector(length=1)
n=length(training_city[,5])
for(i in 1){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = svm(training_city[-k,5] ~ ., data = training_city[-k, -c(1:4)], trace=F) 
    yhat = predict(set_model, newdata=training_city[k, -c(1:4)])
    cvi = cvi + (1 - sum(yhat==training_city[k,5])/length(yhat))
  }
  CV_values_svm[i] = cvi/5
}

CV_values_svm # 13.75% error rate

levels(yhat); levels(training_city[,5])
plyr::count(yhat); plyr::count(training_city[k,5])
# at the last fold CV, as expected, due to lack of data of the new car models,
# our svm model prediction led to more basic models "E" & "S"
# the model has a higher success rate for predicting basic models

# Tune - SVM --------------------------------------------------------------

# initial model: cost = 1, gamma = 0.1, kernel = radial
svm_tune_radial <- tune(svm, Modelvar ~ ., data = training_city[,-c(1:4)],
                        kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))

str(training_city[,-c(1:4)])
levels(training_city$Modelvar)
svm_tune$best.model # cost = 10, gamma = 0.5

CV_values_svm = vector(length=1)
n=length(training_city[,5])
for(i in 1){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = svm(training_city[-k,5] ~ ., data = training_city[-k, -c(1:4)], trace=F,
                    cost = 10, gamma = 0.5) 
    yhat = predict(set_model, newdata=training_city[k, -c(1:4)])
    cvi = cvi + (1 - sum(yhat==training_city[k,5])/length(yhat))
  }
  CV_values_svm[i] = cvi/5
}

CV_values_svm # 4.5% error rate, MAJOR improvement over untuned model

svm_tune_linear <- tune(svm, Modelvar ~  ., data = training_city[,-c(1:4)],
                        kernel="linear", ranges=list(cost=10^(-1:2)))

svm_tune_linear$best.model # cost = 10, gamma = 0.09090909

CV_values_svm = vector(length=1)
n=length(training_city[,5])
for(i in 1){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = svm(training_city[-k,5] ~ ., data = training_city[-k, -c(1:4)], trace=F,
                    kernel = "linear", cost = 10, gamma = 0.09090909) 
    yhat = predict(set_model, newdata=training_city[k, -c(1:4)])
    cvi = cvi + (1 - sum(yhat==training_city[k,5])/length(yhat))
  }
  CV_values_svm[i] = cvi/5
}

CV_values_svm 
# 0.1% error with merged df
# 12.4% error rate, linear kernel perform worst than radial

# Verification vs Testset (Model variants) -------------------------------------------------

svm_city = svm(Modelvar ~ ., data = training_city[,-c(1:4)],
               kernel='radial', cost = 10, gamma = 0.5)
test_output = predict(svm_city, newdata=testing_city[,-c(1:5)])


1-sum(test_output==testing_city[,6])/length(test_output) #26.97% error rate

table(test_output, testing_city[,6]) # compare differences
# Ref: https://afit-r.github.io/svm

# Verification vs Testset (Avg Price of Model variants) -------------------------------------------------

str(training_city)
str(testing_city)

check_price_diff = cbind.data.frame(testing_city$Modelvar, test_output, testing_city$Price)
names(check_price_diff) = c('ActualModel', 'EstModel', 'Price')

check_price_diff_actual = group_by(check_price_diff, ActualModel)
check_price_diff_actual = dplyr::summarise(check_price_diff_actual, Price = mean(Price))

check_price_diff_est = group_by(check_price_diff, EstModel)
check_price_diff_est = dplyr::summarise(check_price_diff_est, Price = mean(Price))

cbind.data.frame(check_price_diff_actual, check_price_diff_est$Price)

check_price_diff %>%
  group_by(ActualModel) %>%
  dplyr::summarise(Price = mean(Price))

check_price_diff %>%
  group_by(EstModel) %>%
  dplyr::summarise(Price = mean(Price))



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


## ## ## ## ## ## ## ##  
# alternative soln to combine levels
# ha <- list(
#  unknown = c("unemployed","unknown","self-employed"),
#  class1  = c("admin.","management")
#)

#for (i in 1:length(ha)) levels(z)[levels(z)%in%ha[[i]]] <- names(ha)[i]
## ## ## ## ## ## ## ##  


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




# Caret - SVM -------------------------------------------------------------

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
