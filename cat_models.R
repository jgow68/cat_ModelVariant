

# Read data from csv ------------------------------------------------------

setwd("~/Github Folder/cat_ModelVariant")

# form training df
data1 = read.csv('May18_V2.csv')
data2 = read.csv('Apr18_V2.csv')
data3 = read.csv('Mar18_V1.csv')

# data1 = read.csv("Feb18_V1.csv")
# data2 = read.csv('Jan18_V1.csv')
# data3 = read.csv('Dec17_V1.csv')
# data4 = read.csv('Nov17_V3.csv')

# form testing df

verify_data = read.csv("Jun18_V2.csv")
# verify_data = read.csv("Mar18_V1.csv")

# search Features for key words and form a new column to merge with dataframe
data1$Features = as.character(data1$Features)
data2$Features = as.character(data2$Features)
data3$Features = as.character(data3$Features)
# data4$Features = as.character(data4$Features)
verify_data$Features = as.character(verify_data$Features)

# Data 1 Search
data1$Airbag = grepl("Airbag", data1$Features)
data1$Leather = grepl("Leather", data1$Features)
data1$Nav = grepl("Nav", data1$Features)
data1$ABS = grepl("ABS", data1$Features)
data1$Sport.Rim = grepl("Sport rim", data1$Features)
data1$Reverse.Camera = grepl("Reverse camera", data1$Features)
data1$Power.Door = grepl("Power Door", data1$Features)
data1$Climate.Control = grepl("Climate control", data1$Features)
data1$Light = grepl("Light", data1$Features)

# Data 2 Search
data2$Airbag = grepl("Airbag", data2$Features)
data2$Leather = grepl("Leather", data2$Features)
data2$Nav = grepl("Nav", data2$Features)
data2$ABS = grepl("ABS", data2$Features)
data2$Sport.Rim = grepl("Sport rim", data2$Features)
data2$Reverse.Camera = grepl("Reverse camera", data2$Features)
data2$Power.Door = grepl("Power Door", data2$Features)
data2$Climate.Control = grepl("Climate control", data2$Features)
data2$Light = grepl("Light", data2$Features)

# Data 3 Search
data3$Airbag = grepl("Airbag", data3$Features)
data3$Leather = grepl("Leather", data3$Features)
data3$Nav = grepl("Nav", data3$Features)
data3$ABS = grepl("ABS", data3$Features)
data3$Sport.Rim = grepl("Sport rim", data3$Features)
data3$Reverse.Camera = grepl("Reverse camera", data3$Features)
data3$Power.Door = grepl("Power Door", data3$Features)
data3$Climate.Control = grepl("Climate control", data3$Features)
data3$Light = grepl("Light", data3$Features)

# Data 4 Search
data4$Airbag = grepl("Airbag", data4$Features)
data4$Leather = grepl("Leather", data4$Features)
data4$Nav = grepl("Nav", data4$Features)
data4$ABS = grepl("ABS", data4$Features)
data4$Sport.Rim = grepl("Sport rim", data4$Features)
data4$Reverse.Camera = grepl("Reverse camera", data4$Features)
data4$Power.Door = grepl("Power Door", data4$Features)
data4$Climate.Control = grepl("Climate control", data4$Features)
data4$Light = grepl("Light", data4$Features)

# Testset Search
verify_data$Airbag = grepl("Airbag", verify_data$Features)
verify_data$Leather = grepl("Leather", verify_data$Features)
verify_data$Nav = grepl("Nav", verify_data$Features)
verify_data$ABS = grepl("ABS", verify_data$Features)
verify_data$Sport.Rim = grepl("Sport rim", verify_data$Features)
verify_data$Reverse.Camera = grepl("Reverse camera", verify_data$Features)
verify_data$Power.Door = grepl("Power Door", verify_data$Features)
verify_data$Climate.Control = grepl("Climate control", verify_data$Features)
verify_data$Light = grepl("Light", verify_data$Features)

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
  data1$Sport.Rim, data1$Reverse.Camera, data1$Power.Door,
  data1$Climate.Control, data1$Light
)

names(training_data1) = c("ID", "Brand", "Model", "Features", "Price",
                         "Modelvar", "MfgYr", "CC", "Transm",
                         "Airbag", "Leather", "Nav", "ABS",
                         "SportRims", "RevCam", "PowDoor",
                         "ClimaCtrl", "Light"
                    )

training_data2 = cbind.data.frame(
  data2$ID, data2$Brand, data2$Model, data2$Features, data2$Price,
  data2$Model.Variant, data2$Mfg_Year, data2$Engine_Capacity, data2$Transmission,
  data2$Airbag, data2$Leather, data2$Nav, data2$ABS, 
  data2$Sport.Rim, data2$Reverse.Camera, data2$Power.Door,
  data2$Climate.Control, data2$Light
)

names(training_data2) = c("ID", "Brand", "Model", "Features", "Price",
                          "Modelvar", "MfgYr", "CC", "Transm",
                          "Airbag", "Leather", "Nav", "ABS",
                          "SportRims", "RevCam", "PowDoor",
                          "ClimaCtrl", "Light"
)

training_data3 = cbind.data.frame(
  data3$ID, data3$Brand, data3$Model, data3$Features, data3$Price,
  data3$Model.Variant, data3$Mfg_Year, data3$Engine_Capacity, data3$Transmission,
  data3$Airbag, data3$Leather, data3$Nav, data3$ABS, 
  data3$Sport.Rim, data3$Reverse.Camera, data3$Power.Door,
  data3$Climate.Control, data3$Light
)

names(training_data3) = c("ID", "Brand", "Model", "Features", "Price",
                          "Modelvar", "MfgYr", "CC", "Transm",
                          "Airbag", "Leather", "Nav", "ABS",
                          "SportRims", "RevCam", "PowDoor",
                          "ClimaCtrl", "Light"
)

training_data4 = cbind.data.frame(
  data4$ID, data4$Brand, data4$Model, data4$Features, data4$Price,
  data4$Model.variant, data4$Mfg_Year, data4$Engine_Capacity, data4$Transmission,
  data4$Airbag, data4$Leather, data4$Nav, data4$ABS, 
  data4$Sport.Rims, data4$Reverse.Camera, data4$Power.Door,
  data4$Climate.Control, data4$Light
)

names(training_data4) = c("ID", "Brand", "Model", "Features", "Price",
                          "Modelvar", "MfgYr", "CC", "Transm",
                          "Airbag", "Leather", "Nav", "ABS",
                          "SportRims", "RevCam", "PowDoor",
                          "ClimaCtrl", "Light"
)

dim(training_data1) # 100k obs
dim(training_data2) # 100k obs
dim(training_data3) # 100k obs
# dim(training_data4) #109,591 obs

# 1st case: choose most recent data set for training
# training_data = training_data1

# 2nd case: merge historical datasets by outer join 2 df
# consider using data.table for faster computation

# Remove duplicates that exist in data set right from the start

training_data1 = training_data1[!duplicated(training_data1$ID), ]
training_data2 = training_data2[!duplicated(training_data2$ID), ]
training_data3 = training_data3[!duplicated(training_data3$ID), ]

# training_data_merged = merge(training_data1, training_data2, all=TRUE) # use left join to prioritize newer datasets
# training_data = merge(training_data, training_data3, all=TRUE)
# training_data = merge(training_data, training_data4, all=TRUE)

rm(training_data) # reset training data

# Merge data set for training (Can be improved)

training_data_merged = dplyr::left_join(training_data1, training_data2)

head(training_data_merged[duplicated(training_data_merged$ID), ])

ID_init = training_data_merged$ID
training_data_init = dplyr::filter(training_data2, !(ID %in% ID_init))

training_data = merge(training_data_merged, training_data_init, all=TRUE)

head(training_data[duplicated(training_data$ID), ])

training_data = merge(training_data, training_data2[!ID_init, ], all=TRUE)

head(training_data[duplicated(training_data$ID), ])

# now merge the 3rd dataset
ID_init = training_data$ID
training_data_init = dplyr::filter(training_data3, !(ID %in% ID_init))

training_data = merge(training_data, training_data_init, all=TRUE)
head(training_data[duplicated(training_data$ID), ])

dim(training_data)
# merge Mar-May'18 ~210k obs

# re-factorize data

training_data$Brand = as.factor(toupper(trimws(training_data$Brand)))
training_data$Modelvar = as.factor(toupper(trimws(training_data$Modelvar)))
training_data$Model = as.factor(toupper(trimws(training_data$Model)))
training_data$Transm = as.factor(toupper(trimws(training_data$Transm)))
training_data$MfgYr = as.factor(toupper(trimws(training_data$MfgYr)))

training_data$CC = as.numeric(training_data$CC)
training_data$Price = as.numeric(gsub('[,]', '', training_data$Price))

# filter out empty cells & NAs
training_data = training_data[complete.cases(training_data[ , -4]), ]

training_data = dplyr::filter(training_data, Brand != "" & Model != "" &
                                !(Modelvar %in% c("", "#N/A", 0, "-")) & 
                                !(MfgYr %in% c("", "1995 OR OLDER")) &
                                Transm != "")


training_data = droplevels(training_data)

training_data$CC_adj = ifelse(training_data$CC<100, training_data$CC*100,
                              ifelse(training_data$CC<10000, training_data$CC, training_data$CC))

training_data = dplyr::filter(training_data, !is.na(CC_adj) & CC_adj < 10000)
# scale data at respective make level


# Prepare testing dataset
testing_data = cbind.data.frame(
  verify_data$ID, verify_data$Brand, verify_data$Model, verify_data$Features, verify_data$Price,
  verify_data$Model.Variant, verify_data$Mfg_Year, verify_data$Engine_Capacity, verify_data$Transmission,
  verify_data$Airbag, verify_data$Leather, verify_data$Nav, verify_data$ABS, 
  verify_data$Sport.Rim, verify_data$Reverse.Camera, verify_data$Power.Door,
  verify_data$Climate.Control, verify_data$Light
)

names(testing_data) = c("ID", "Brand", "Model", "Features", "Price",
                         "Modelvar", "MfgYr", "CC", "Transm",
                         "Airbag", "Leather", "Nav", "ABS",
                         "SportRims", "RevCam", "PowDoor",
                         "ClimaCtrl", "Light"
)

testing_data = testing_data[!duplicated(testing_data$ID), ]

testing_data$Brand = as.factor(toupper(trimws(testing_data$Brand)))
testing_data$Modelvar = as.factor(toupper(trimws(testing_data$Modelvar)))
testing_data$Model = as.factor(toupper(trimws(testing_data$Model)))
testing_data$Transm = as.factor(toupper(trimws(testing_data$Transm)))
testing_data$MfgYr = as.factor(toupper(trimws(testing_data$MfgYr)))

testing_data$Price = as.numeric(gsub('[,]', '', testing_data$Price))

testing_data = testing_data[complete.cases(testing_data[ , -4]), ]

testing_data = dplyr::filter(testing_data, Brand != "" & Model != "" &
                               !(Modelvar %in% c("", "#N/A", 0, "-")) &
                               !(MfgYr %in% c("", "1995 OR OLDER")) &
                               Transm != "")

testing_data = droplevels(testing_data)

testing_data$CC_adj = ifelse(testing_data$CC<100, testing_data$CC*100,
                             ifelse(testing_data$CC<10000, testing_data$CC, testing_data$CC))

testing_data = dplyr::filter(testing_data, !is.na(CC_adj) & CC_adj < 10000)

colnames(training_data) == colnames(testing_data)

# Chevrolet: prep data ------------------------------------

train_che = dplyr::filter(training_data, Brand=="CHEVROLET" &
                            !Modelvar %in% c('-', '0', '#N/A')
                          )
summary(train_che[, -4]) # summary without the long list of features text

test_che = dplyr::filter(testing_data, Brand=="CHEVROLET" & 
                           !Modelvar %in% c('-', '0', '#N/A')
                         )
summary(test_che[, -4])

train_che$CC_scl = as.numeric(scale(train_che$CC_adj))
test_che$CC_scl = as.numeric(scale(test_che$CC_adj))

plyr::count(train_che$Modelvar)
length(train_che$Modelvar)

# filter training set only for variants with size >20
train_che = train_che %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>20) %>%
  as.data.frame()

train_che = droplevels(train_che)
test_che = droplevels(test_che)

# standardize Variant levels
che_var = unique(c(levels(train_che$Modelvar),levels(test_che$Modelvar)))
train_che$Modelvar = factor(train_che$Modelvar, levels = che_var)
test_che$Modelvar = factor(test_che$Modelvar, levels = che_var)

summary(train_che$Modelvar)
summary(test_che$Modelvar)

# standardize Model levels
che_mdl = unique(c(levels(train_che$Model),levels(test_che$Model)))
train_che$Model = factor(train_che$Model, levels = che_mdl)
test_che$Model = factor(test_che$Model, levels = che_mdl)

summary(train_che$Model)
summary(test_che$Model)

# standardize MfgYr levels
che_yr = unique(c(levels(train_che$MfgYr), levels(test_che$MfgYr)))
train_che$MfgYr = factor(train_che$MfgYr, levels = che_yr)
test_che$MfgYr = factor(test_che$MfgYr, levels = che_yr)

summary(train_che$MfgYr)
summary(test_che$MfgYr)

# Chevrolet: id pred ------------------------------------------

# auto select predictors to drop if all are TRUE/FALSE

for (i in 9:(length(colnames(train_che))-2)){
  test = plyr::count(train_che, colnames(train_che)[i])
  if (length(test$freq) == 1){ 
    # alternative we can add: if length(test$freq) > 1, then test$freq[1]/test$freq[2] < 5% then remove
    to_drop = c(paste(colnames(train_che)[i]), to_drop)
  }
}

drops = c(to_drop, "ID", "Brand", "Features", "Price", "CC", "CC_adj")

# drop all non-relevant predictors
train_che = train_che[, !names(train_che) %in% drops]
test_che = test_che[, !names(test_che) %in% drops]

colnames(train_che) == colnames(test_che)
colnames(train_che)
str(train_che)

# Chevrolet: SVM ----------------------------------------------------------

system.time(tune_radial_che <- tune(svm, Modelvar ~ ., data = train_che,
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 200 obs, 17- 6 var

# 200 obs, 19-6 variables, Transm removed
# user  system elapsed 
# 3.69    0.08    3.82 

tune_radial_che$best.parameters
# excl. Transm, Rims, Climactrl, cost=1, gamma=0.1
# incl. manf yr, CC cost=1, gamma=0.1

svm_che = svm(Modelvar ~., data = train_che,
              kernel="radial", cost=tune_radial_che$best.parameters$cost,
              gamma=tune_radial_che$best.parameters$gamma, trace=F
)

svm_est_che = predict(svm_che, newdata=test_che)

che_est_lvl = unique(c(levels(svm_est_che), levels(test_che$Modelvar)))
test_che$Modelvar = factor(test_che$Modelvar, levels = che_est_lvl)
svm_est_che = factor(svm_est_che, levels = che_est_lvl)


1-sum(test_che$Modelvar==svm_est_che)/nrow(test_che)

# Error info:
# 30% (excl. rims, transm, climactrl)
# 20% (excl. Transm, Rims, Climactrl) - reason training missed out several Model Grps
# 10% (incl MfgYr, CC) excl Model Group

# Note: if training set do not have particular Model Grp, wont be able to x-check the accuracy

che_tbl = table(test_che$Modelvar, svm_est_che)
write.csv(che_tbl,"che.csv")

# Suzuki: prep data -------------------------------------------------------

train_szk = dplyr::filter(training_data, Brand=="SUZUKI" &
                            !(Modelvar %in% c('', '-', '0', '#N/A')) &
                            !is.na(Modelvar))
summary(train_szk[, -4])

test_szk = dplyr::filter(testing_data, Brand=="SUZUKI" & 
                           !(Modelvar %in% c('', '-', '0', '#N/A')) &
                           !is.na(Modelvar))
summary(test_szk[, -4])

summary(train_szk$CC_adj)
summary(test_szk$CC_adj)

train_szk$CC_scl = as.numeric(scale(train_szk$CC_adj))
test_szk$CC_scl = as.numeric(scale(test_szk$CC_adj))

train_szk = train_szk %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>20) %>%
  as.data.frame()

train_szk = droplevels(train_szk)
test_szk = droplevels(test_szk)

# standardize Variant levels
szk_var = unique(c(levels(train_szk$Modelvar),levels(test_szk$Modelvar)))
train_szk$Modelvar = factor(train_szk$Modelvar, levels = szk_var)
test_szk$Modelvar = factor(test_szk$Modelvar, levels = szk_var)

summary(train_szk$Modelvar)
summary(test_szk$Modelvar)

# standardize Model levels
szk_mdl = unique(c(levels(train_szk$Model),levels(test_szk$Model)))
train_szk$Model = factor(train_szk$Model, levels = szk_mdl)
test_szk$Model = factor(test_szk$Model, levels = szk_mdl)

summary(train_szk$Model)
summary(test_szk$Model)

# standardize MfgYr levels
szk_yr = unique(c(levels(train_szk$MfgYr), levels(test_szk$MfgYr)))
train_szk$MfgYr = factor(train_szk$MfgYr, levels = szk_yr)
test_szk$MfgYr = factor(test_szk$MfgYr, levels = szk_yr)

summary(train_szk$MfgYr)
summary(test_szk$MfgYr)

# Suzuki: id pred ---------------------------------------------------------

for (i in 9:(length(colnames(train_szk))-2)){
  test = plyr::count(train_szk, colnames(train_szk)[i])
  if (length(test$freq) == 1){ 
    # alternative we can add: if length(test$freq) > 1, then test$freq[1]/test$freq[2] < 5% then remove
    to_drop = c(paste(colnames(train_szk)[i]), to_drop)
  }
}

drops = c(to_drop, "ID", "Brand", "Features", "Price", "CC", "CC_adj")

# drop all non-relevant predictors
train_szk = train_szk[, !names(train_szk) %in% drops]
test_szk = test_szk[, !names(test_szk) %in% drops]

colnames(train_szk) == colnames(test_szk)
colnames(train_szk)


# Suzuki: SVM -------------------------------------------------------------

system.time(tune_radial_szk <- tune(svm, Modelvar ~ ., data = train_szk,
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

dim(train_szk)
# 555 obs, 20-6 variables
# user  system elapsed 
# 14.46    0.25   15.07

tune_radial_szk$best.parameters 
# incl. manf yr, CC cost=100, gamma=0.1

# Model Group NA
svm_szk = svm(Modelvar ~., data = train_szk,
              kernel="radial", cost=tune_radial_szk$best.parameters$cost,
              gamma=tune_radial_szk$best.parameters$gamma, trace=F
)

svm_est_szk = predict(svm_szk, newdata=test_szk)

szk_est_lvl = unique(c(levels(svm_est_szk), levels(test_szk$Modelvar)))
test_szk$Modelvar = factor(test_szk$Modelvar, levels = szk_est_lvl)
svm_est_szk = factor(svm_est_szk, levels = szk_est_lvl)

1-sum(test_szk$Modelvar==svm_est_szk)/nrow(test_szk)

szk_tbl = table(test_szk$Modelvar, svm_est_szk)
write.csv(szk_tbl,"szk.csv")

c(colnames(train_szk), dim(train_szk))

# Others TBD: Peugeot, BMW, Isuzu ------------------------------------------------------------------
# Peugeot
# BMW
# Isuzu

# Mazda: prep data --------------------------------------------------------

train_mzd = dplyr::filter(training_data, Brand=="Mazda")
head(plyr::count(train_mzd$Modelvar))

test_mzd = dplyr::filter(testing_data, Brand=="Mazda" & 
                            !Modelvar %in% c('', '-', '0', '#N/A'))
head(plyr::count(test_mzd$Modelvar))

# clean up spaces / upper-lower char and re-factor
train_mzd$Modelvar = as.factor(toupper(trimws(train_mzd$Modelvar)))
test_mzd$Modelvar = as.factor(toupper(trimws(test_mzd$Modelvar)))

plyr::count(train_mzd$CC_adj)
plyr::count(test_mzd$CC_adj)

train_mzd$CC_scl = as.numeric(scale(train_mzd$CC_adj))
test_mzd$CC_scl = as.numeric(scale(test_mzd$CC_adj))

# filter for variants with size > 30
train_mzd = train_mzd %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>30) %>%
  as.data.frame()

test_mzd = dplyr::filter(test_mzd, Transm != "")

train_mzd = droplevels(train_mzd)
test_mzd = droplevels(test_mzd)

# standardize Variant levels
mzd_var = unique(c(levels(train_mzd$Modelvar),levels(test_mzd$Modelvar)))
train_mzd$Modelvar = factor(train_mzd$Modelvar, levels = mzd_var)
test_mzd$Modelvar = factor(test_mzd$Modelvar, levels = mzd_var)
summary(train_mzd$Modelvar)
summary(test_mzd$Modelvar)

# standardize Model levels
mzd_mdl = unique(c(levels(train_mzd$Model),levels(test_mzd$Model)))
train_mzd$Model = factor(train_mzd$Model, levels = mzd_mdl)
test_mzd$Model = factor(test_mzd$Model, levels = mzd_mdl)

summary(train_mzd$Model)
summary(test_mzd$Model)

# standardize MfgYr levels
mzd_yr = unique(c(levels(train_mzd$MfgYr), levels(test_mzd$MfgYr)))
train_mzd$MfgYr = factor(train_mzd$MfgYr, levels = mzd_yr)
test_mzd$MfgYr = factor(test_mzd$MfgYr, levels = mzd_yr)

summary(train_mzd$MfgYr)
summary(test_mzd$MfgYr)


# Mazda: id pred ----------------------------------------------------------

plyr::count(train_mzd, "Transm") # 1 manual
plyr::count(train_mzd, "Airbag")
plyr::count(train_mzd, "Leather")
plyr::count(train_mzd, "Nav")
plyr::count(train_mzd, "ABS")
plyr::count(train_mzd, "SportRims")
plyr::count(train_mzd, "RevCam")
plyr::count(train_mzd, "PowDoor") # weird some units have power door?
plyr::count(train_mzd, "ClimaCtrl")
plyr::count(train_mzd, "Light")

colnames(train_mzd) == colnames(test_mzd)
colnames(train_mzd)
str(train_mzd)

# Mazda: SVM --------------------------------------------------------------

system.time(tune_radial_mzd <- tune(svm, Modelvar ~ ., data = train_mzd[,-c(1:3, 4:5, 8, 19)],
                                     kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 700 obs, 20-6 variables
# user  system elapsed 
# 23.23    0.01   23.51

tune_radial_mzd$best.parameters

# incl. manf yr, CC cost=10, gamma=0.1

# Model Group NA
svm_mzd = svm(Modelvar ~., data = train_mzd[,-c(1:3, 4:5, 8, 19)],
               kernel="radial", cost=tune_radial_mzd$best.parameters$cost,
               gamma=tune_radial_mzd$best.parameters$gamma, trace=F
)

svm_est_mzd = predict(svm_mzd, newdata=test_mzd[,-c(1:3, 4:5, 8, 19)])

mzd_var_lvl = unique(c(levels(svm_est_mzd), levels(test_mzd$Modelvar)))
test_mzd$Modelvar = factor(test_mzd$Modelvar, levels = mzd_var_lvl)
svm_est_mzd = factor(svm_est_mzd, levels = mzd_var_lvl)

1-sum(test_mzd[,6]==svm_est_mzd)/nrow(test_mzd)

# Error info:
# 33% (incl MfgYr, CC) excl Model Group

mzd_tbl = table(test_mzd$Modelvar, svm_est_mzd)
write.csv(mzd_tbl,"mzd.csv")

# Ford: prep data  --------------------------------------------------------------------

train_ford = dplyr::filter(training_data, Brand=="Ford")

# test
test_ford[complete.cases(test_ford[, ])]
test_ford = dplyr::filter(testing_data, Brand=="Ford" & 
                            !Modelvar %in% c('', '-', '0', '#N/A') &
                            !is.na(Modelvar) &
                            !is.na(MfgYr))

# clean up spaces / upper-lower char and re-factor
train_ford$Modelvar = as.factor(toupper(trimws(train_ford$Modelvar)))
test_ford$Modelvar = as.factor(toupper(trimws(test_ford$Modelvar)))

summary(train_ford$CC_adj)
summary(test_ford$CC_adj)

train_ford$CC_scl = as.numeric(scale(train_ford$CC_adj))
test_ford$CC_scl = as.numeric(scale(test_ford$CC_adj))

# filter for variants with size > 30
train_ford = train_ford %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>30) %>%
  as.data.frame()

train_ford = droplevels(train_ford)
test_ford = droplevels(test_ford)

# standardize Variant levels
ford_var = unique(c(levels(train_ford$Modelvar),levels(test_ford$Modelvar)))
train_ford$Modelvar = factor(train_ford$Modelvar, levels = ford_var)
test_ford$Modelvar = factor(test_ford$Modelvar, levels = ford_var)
summary(train_ford$Modelvar)
summary(test_ford$Modelvar)

# standardize Model levels
ford_mdl = unique(c(levels(train_ford$Model),levels(test_ford$Model)))
train_ford$Model = factor(train_ford$Model, levels = ford_mdl)
test_ford$Model = factor(test_ford$Model, levels = ford_mdl)

summary(train_ford$Model)
summary(test_ford$Model)
test_ford = dplyr::filter(test_ford, !is.na(Model))

# standardize MfgYr levels
ford_yr = unique(levels(train_ford$MfgYr), levels(test_ford$MfgYr))
train_ford$MfgYr = factor(train_ford$MfgYr, levels = ford_yr)
test_ford$MfgYr = factor(test_ford$MfgYr, levels = ford_yr)

summary(train_ford$MfgYr)
summary(test_ford$MfgYr)


# Ford: identify predictors -----------------------------------------------

plyr::count(train_ford, "Transm")
plyr::count(train_ford, "Airbag")
plyr::count(train_ford, "Leather")
plyr::count(train_ford, "Nav")
plyr::count(train_ford, "ABS")
plyr::count(train_ford, "SportRims")
plyr::count(train_ford, "RevCam")
plyr::count(train_ford, "PowDoor") # weird some units have power door?
plyr::count(train_ford, "ClimaCtrl")
plyr::count(train_ford, "Light")

colnames(train_ford) == colnames(test_ford)
colnames(train_ford)
str(train_ford)

# Ford: SVM ---------------------------------------------------------------

library(e1071)
system.time(tune_radial_ford <- tune(svm, Modelvar ~ ., data = train_ford[,-c(1:2, 4:5, 8, 19)],
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 700 obs, 20-6 variables
# user  system elapsed 
# 23.23    0.01   23.51

tune_radial_ford$best.parameters

# incl. manf yr, CC cost=100, gamma=0.1

svm_ford = svm(Modelvar ~., data = train_ford[,-c(1:2, 4:5, 8, 19)],
              kernel="radial", cost=tune_radial_ford$best.parameters$cost,
              gamma=tune_radial_ford$best.parameters$gamma, trace=F
)

svm_est_ford = predict(svm_ford, newdata=test_ford[,-c(1:2, 4:5, 8, 19)])
ford_var_lvl = unique(c(levels(test_ford$Modelvar), levels(svm_est_ford)))
test_ford$Modelvar = factor(test_ford$Modelvar, levels = ford_var_lvl)
svm_est_ford = factor(svm_est_ford, levels = ford_var_lvl)

1-sum(test_ford$Modelvar==svm_est_ford)/nrow(test_ford)

# Error info:
# 29% (incl MfgYr, CC)

ford_tbl = table(test_ford$Modelvar, svm_est_ford)
write.csv(ford_tbl,"ford.csv")

# Mitsubishi: prep data ---------------------------------------------------------

train_mit = dplyr::filter(training_data, Brand=="Mitsubishi")

test_mit = dplyr::filter(testing_data, Brand=="Mitsubishi" & 
                           !Modelvar %in% c('', '-', '0', '#N/A') &
                           !is.na(Model))

# clean up spaces / upper-lower char and re-factor
train_mit$Modelvar = as.factor(toupper(trimws(train_mit$Modelvar)))
test_mit$Modelvar = as.factor(toupper(trimws(test_mit$Modelvar)))

summary(train_mit$CC_adj)
summary(test_mit$CC_adj)

train_mit$CC_scl = as.numeric(scale(train_mit$CC_adj))
test_mit$CC_scl = as.numeric(scale(test_mit$CC_adj))

# filter for variants with size > 30
train_mit = train_mit %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>30) %>%
  as.data.frame()

train_mit = droplevels(train_mit)
test_mit = droplevels(test_mit)

# standardize Model levels
mit_mdl = unique(c(levels(train_mit$Model),levels(test_mit$Model)))
train_mit$Model = factor(train_mit$Model, levels = mit_mdl)
test_mit$Model = factor(test_mit$Model, levels = mit_mdl)

summary(train_mit$Model)
summary(test_mit$Model)

# standardize MfgYr levels
mit_yr = unique(c(levels(train_mit$MfgYr), levels(test_mit$MfgYr)))
train_mit$MfgYr = factor(train_mit$MfgYr, levels = mit_yr)
test_mit$MfgYr = factor(test_mit$MfgYr, levels = mit_yr)

summary(train_mit$MfgYr)
summary(test_mit$MfgYr)

# Mitsubishi: identify predictors -----------------------------------------

plyr::count(train_mit, "Transm")
plyr::count(train_mit, "Airbag")
plyr::count(train_mit, "Leather")
plyr::count(train_mit, "Nav")
plyr::count(train_mit, "ABS")
plyr::count(train_mit, "SportRims")
plyr::count(train_mit, "RevCam")
plyr::count(train_mit, "PowDoor") # weird some units have power door?
plyr::count(train_mit, "ClimaCtrl")
plyr::count(train_mit, "Light")

colnames(train_mit) == colnames(test_mit)
colnames(train_mit)
str(train_mit)

# Mitsubishi: SVM ---------------------------------------------------------

system.time(tune_radial_mit <- tune(svm, Modelvar ~ ., data = train_mit[,-c(1:2, 4:5, 8, 19)],
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 932 obs, 20-6 variables
# user  system elapsed 
# 41.31    0.16   42.37 

tune_radial_mit$best.parameters

# incl. manf yr, CC cost=10, gamma=0.1

svm_mit = svm(Modelvar ~., data = train_mit[,-c(1:2, 4:5, 8, 19)],
              kernel="radial", cost=tune_radial_mit$best.parameters$cost,
              gamma=tune_radial_mit$best.parameters$gamma, trace=F
)

svm_est_mit = predict(svm_mit, newdata=test_mit[,-c(1:2, 4:5, 8, 19)])

mit_var = unique(c(levels(svm_est_mit),levels(test_mit$Modelvar)))
svm_est_mit = factor(svm_est_mit, levels = mit_var)
test_mit$Modelvar = factor(test_mit$Modelvar, levels = mit_var)

1-sum(test_mit$Modelvar==svm_est_mit)/nrow(test_mit)

# Error info:
# 21% (incl MfgYr, CC)

mit_tbl = table(test_mit$Modelvar, svm_est_mit)
write.csv(mit_tbl,"mit.csv")

# Kia: All ----------------------------------------------------------------

train_kia = dplyr::filter(training_data, Brand=="Kia")
test_kia = dplyr::filter(testing_data, Brand=="Kia" & 
                           !Modelvar %in% c('', '-', '0', '#N/A'))

# clean up spaces / upper-lower char and re-factor
train_kia$Modelvar = as.factor(toupper(trimws(train_kia$Modelvar)))
test_kia$Modelvar = as.factor(toupper(trimws(test_kia$Modelvar)))

summary(train_kia$CC_adj)
summary(test_kia$CC_adj)

train_kia$CC_scl = as.numeric(scale(train_kia$CC_adj))
test_kia$CC_scl = as.numeric(scale(test_kia$CC_adj))

# filter for variants with size > 30
train_kia = train_kia %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>30) %>%
  as.data.frame()

train_kia = droplevels(train_kia)
test_kia = droplevels(test_kia)

# standardize Model levels
kia_mdl = unique(c(levels(train_kia$Model),levels(test_kia$Model)))
train_kia$Model = factor(train_kia$Model, levels = kia_mdl)
test_kia$Model = factor(test_kia$Model, levels = kia_mdl)

summary(train_kia$Model)
summary(test_kia$Model)

# standardize MfgYr levels
kia_yr = unique(c(levels(train_kia$MfgYr), levels(test_kia$MfgYr)))
train_kia$MfgYr = factor(train_kia$MfgYr, levels = kia_yr)
test_kia$MfgYr = factor(test_kia$MfgYr, levels = kia_yr)

summary(train_kia$MfgYr)
summary(test_kia$MfgYr)

# Kia: identify predictors ------------------------------------------------

plyr::count(train_kia, "Transm")
plyr::count(train_kia, "Airbag")
plyr::count(train_kia, "Leather")
plyr::count(train_kia, "Nav")
plyr::count(train_kia, "ABS")
plyr::count(train_kia, "SportRims")
plyr::count(train_kia, "RevCam")
plyr::count(train_kia, "PowDoor") # weird some units have power door?
plyr::count(train_kia, "ClimaCtrl")
plyr::count(train_kia, "Light")

# drop non-useful predictors
drops = c("CC", "TouchScreen")
train_kia = train_kia[, !names(train_kia) %in% drops]
test_kia = test_kia[, !names(test_kia) %in% drops]

colnames(train_kia) == colnames(test_kia)
colnames(train_kia)
str(train_kia)


# Kia: SVM ----------------------------------------------------------------

system.time(tune_radial_kia <- tune(svm, Modelvar ~ ., data = train_kia[,-c(1:2, 4:5, 17)],
                                   kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 1,213 obs, 19-5 variables
# user  system elapsed 
# 71.20    0.05   72.37

tune_radial_kia$best.parameters

# incl. manf yr, CC cost=100, gamma=0.1

svm_kia = svm(Modelvar ~., data = train_kia[,-c(1:2, 4:5, 17)],
             kernel="radial", cost=tune_radial_kia$best.parameters$cost,
             gamma=tune_radial_kia$best.parameters$gamma, trace=F
)

svm_est_kia = predict(svm_kia, newdata=test_kia[,-c(1:2, 4:5, 17)])

kia_var = unique(c(levels(svm_est_kia),levels(test_kia$Modelvar)))
svm_est_kia = factor(svm_est_kia, levels = kia_var)
test_kia$Modelvar = factor(test_kia$Modelvar, levels = kia_var)

1-sum(test_kia$Modelvar==svm_est_kia)/nrow(test_kia)

# Error info:
# 11% (incl MfgYr, CC)

kia_tbl = table(test_kia$Modelvar, svm_est_kia)
write.csv(kia_tbl,"kia.csv")

# Hyundai: All ------------------------------------------------------------

train_hy = dplyr::filter(training_data, Brand=="Hyundai")
test_hy = dplyr::filter(testing_data, Brand=="Hyundai" & 
                          !Modelvar %in% c('', '-', '0', '#N/A'))


# clean up spaces / upper-lower char and re-factor
train_hy$Modelvar = as.factor(toupper(trimws(train_hy$Modelvar)))
test_hy$Modelvar = as.factor(toupper(trimws(test_hy$Modelvar)))

summary(train_hy$CC_adj)
summary(test_hy$CC_adj)

train_hy$CC_scl = as.numeric(scale(train_hy$CC_adj))
test_hy$CC_scl = as.numeric(scale(test_hy$CC_adj))

# filter for variants with size > 30

train_hy = train_hy %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>30) %>%
  as.data.frame()

train_hy = droplevels(train_hy)
test_hy = droplevels(test_hy)

# standardize Variant levels
hy_var = unique(c(levels(train_hy$Modelvar),levels(test_hy$Modelvar)))
train_hy$Modelvar = factor(train_hy$Modelvar, levels = hy_var)
test_hy$Modelvar = factor(test_hy$Modelvar, levels = hy_var)
summary(train_hy$Modelvar)
summary(test_hy$Modelvar)

# standardize Model levels
hy_mdl = unique(levels(train_hy$Model),levels(test_hy$Model))
train_hy$Model = factor(train_hy$Model, levels = hy_mdl)
test_hy$Model = factor(test_hy$Model, levels = hy_mdl)

summary(train_hy$Model)
summary(test_hy$Model)

# standardize MfgYr levels
hy_yr = unique(levels(train_hy$MfgYr), levels(test_hy$MfgYr))
train_hy$MfgYr = factor(train_hy$MfgYr, levels = hy_yr)
test_hy$MfgYr = factor(test_hy$MfgYr, levels = hy_yr)

summary(train_hy) # check for NAs
summary(test_hy) # check for NAs

# Hyundai: identify predictors --------------------------------------------

plyr::count(train_hy, "Transm")
plyr::count(train_hy, "Airbag")
plyr::count(train_hy, "Leather")
plyr::count(train_hy, "Nav")
plyr::count(train_hy, "ABS")
plyr::count(train_hy, "SportRims")
plyr::count(train_hy, "RevCam")
plyr::count(train_hy, "PowDoor") # weird some units have power door?
plyr::count(train_hy, "TouchScreen") # all N/A
plyr::count(train_hy, "ClimaCtrl")

test_search = grepl("leather", test_hy$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

test_search = grepl("light", test_hy$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

# drop non-useful predictors
drops = c("CC", "TouchScreen")
train_hy = train_hy[, !names(train_hy) %in% drops]
test_hy = test_hy[, !names(test_hy) %in% drops]

leather_hyTr = grepl("leather", train_hy$Features, ignore.case=TRUE)
leather_hyTt = grepl("leather", test_hy$Features, ignore.case=TRUE)

light_hyTr = grepl("light", train_hy$Features, ignore.case=TRUE)
light_hyTt = grepl("light", test_hy$Features, ignore.case=TRUE)

train_hy$Leather = leather_hyTr
test_hy$Leather = leather_hyTt

train_hy$Light = light_hyTr
test_hy$Light = light_hyTt

colnames(train_hy) == colnames(test_hy)

colnames(train_hy)
str(train_hy)

# Hyundai: SVM ------------------------------------------------------------

library(e1071)
system.time(tune_radial_hy <- tune(svm, Modelvar ~ ., data = train_hy[,-c(1:2, 4:5, 17)],
                                   kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 1,636 obs, 19-5 variables
# user  system elapsed 
# 161.22    0.15  167.92 ~2.7 min

tune_radial_hy$best.parameters

# incl. manf yr, CC cost=100, gamma=0.1

svm_hy = svm(Modelvar ~., data = train_hy[,-c(1:2, 4:5, 17)],
             kernel="radial", cost=tune_radial_hy$best.parameters$cost,
             gamma=tune_radial_hy$best.parameters$gamma, trace=F
)

svm_est_hy = predict(svm_hy, newdata=test_hy[,-c(1:2, 4:5, 17)])

1-sum(test_hy[,6]==svm_est_hy)/nrow(test_hy)

# Error info:
# 0.4% (incl MfgYr, CC)

hy_tbl = table(test_hy[,6], svm_est_hy)
write.csv(hy_tbl,"hy.csv")

# VW: All -----------------------------------------------------------------

train_vw = dplyr::filter(training_data, Brand=="Volkswagen")
head(plyr::count(train_vw$Modelvar))

test_vw = dplyr::filter(testing_data, Brand=="Volkswagen" & 
                           !Modelvar %in% c('', '-', '0', '#N/A'))
head(plyr::count(test_vw$Modelvar))

# clean up spaces / upper-lower char and re-factor
train_vw$Modelvar = as.factor(toupper(trimws(train_vw$Modelvar)))
test_vw$Modelvar = as.factor(toupper(trimws(test_vw$Modelvar)))

plyr::count(train_vw$CC_adj)
plyr::count(test_vw$CC_adj)

train_vw$CC_scl = as.numeric(scale(train_vw$CC_adj))
test_vw$CC_scl = as.numeric(scale(test_vw$CC_adj))

# filter for variants with size > 30
library(dplyr)
train_vw = train_vw %>% 
  group_by(Modelvar) %>%
  filter(n()>30) %>%
  as.data.frame()

test_vw = test_vw %>% 
  group_by(Modelvar) %>%
  filter(n()>30) %>%
  as.data.frame()

train_vw = droplevels(train_vw)
test_vw = droplevels(test_vw)

# standardize Variant levels
vw_var = unique(c(levels(train_vw$Modelvar),levels(test_vw$Modelvar)))
train_vw$Modelvar = factor(train_vw$Modelvar, levels = vw_var)
test_vw$Modelvar = factor(test_vw$Modelvar, levels = vw_var)
summary(train_vw$Modelvar)
summary(test_vw$Modelvar)

# standardize Model levels
vw_mdl = unique(levels(train_vw$Model),levels(test_vw$Model))
train_vw$Model = factor(train_vw$Model, levels = vw_mdl)
test_vw$Model = factor(test_vw$Model, levels = vw_mdl)

summary(train_vw$Model)
test_vw = dplyr::filter(test_vw, !is.na(Model))
summary(test_vw$Model)

# standardize MfgYr levels
vw_yr = unique(levels(train_vw$MfgYr), levels(test_vw$MfgYr))
train_vw$MfgYr = factor(train_vw$MfgYr, levels = vw_yr)
test_vw$MfgYr = factor(test_vw$MfgYr, levels = vw_yr)

summary(train_vw) # check for NAs
summary(test_vw) # check for NAs


# VW: identify predictors -------------------------------------------------

plyr::count(train_vw, "Transm") # no manual
plyr::count(train_vw, "Airbag")
plyr::count(train_vw, "Leather")
plyr::count(train_vw, "Nav")
plyr::count(train_vw, "ABS")
plyr::count(train_vw, "SportRims")
plyr::count(train_vw, "RevCam")
plyr::count(train_vw, "PowDoor") # weird some units have power door?
plyr::count(train_vw, "TouchScreen") # all N/A
plyr::count(train_vw, "ClimaCtrl")

test_search = grepl("leather", test_vw$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

test_search = grepl("light", test_vw$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

# drop non-useful predictors
drops = c("Transm", "CC", "TouchScreen")
train_vw = train_vw[, !names(train_vw) %in% drops]
test_vw = test_vw[, !names(test_vw) %in% drops]

leather_vwTr = grepl("leather", train_vw$Features, ignore.case=TRUE)
leather_vwTt = grepl("leather", test_vw$Features, ignore.case=TRUE)

light_vwTr = grepl("light", train_vw$Features, ignore.case=TRUE)
light_vwTt = grepl("light", test_vw$Features, ignore.case=TRUE)

train_vw$Leather = leather_vwTr
test_vw$Leather = leather_vwTt

train_vw$Light = light_vwTr
test_vw$Light = light_vwTt

colnames(train_vw) == colnames(test_vw)

colnames(train_vw)
str(train_vw)


# VW: SVM -----------------------------------------------------------------

library(e1071)
system.time(tune_radial_vw <- tune(svm, Modelvar ~ ., data = train_vw[,-c(1:2, 4:5, 17)],
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 1,354 obs, 19-5 variables
# user  system elapsed 
# 85.16    0.07   88.73 ~ 1.5 mins

tune_radial_vw$best.parameters

# incl. manf yr, CC cost=10, gamma=0.1

svm_vw = svm(Modelvar ~., data = train_vw[,-c(1:2, 4:5, 17)],
              kernel="radial", cost=tune_radial_vw$best.parameters$cost,
              gamma=tune_radial_vw$best.parameters$gamma, trace=F
)

svm_est_vw = predict(svm_vw, newdata=test_vw[,-c(1:2, 4:5, 17)])

1-sum(test_vw[,6]==svm_est_vw)/nrow(test_vw)

# Error info:
# 2.5% (incl MfgYr, CC)

vw_tbl = table(test_vw[,6], svm_est_vw)
write.csv(vw_tbl,"vw.csv")

# Nissan: All -------------------------------------------------------------

train_nis = dplyr::filter(training_data, Brand=="NISSAN")
summary(train_nis[, -4])

test_nis = dplyr::filter(testing_data, Brand=="NISSAN")
summary(test_nis[, -4])

train_nis$CC_scl = as.numeric(scale(train_nis$CC_adj))
test_nis$CC_scl = as.numeric(scale(test_nis$CC_adj))

# filter training set only for variants with size >20
train_nis = train_nis %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>20) %>%
  as.data.frame()

train_nis = droplevels(train_nis)
test_nis = droplevels(test_nis)

# standardize Variant levels
nis_var = unique(c(levels(train_nis$Modelvar),levels(test_nis$Modelvar)))
train_nis$Modelvar = factor(train_nis$Modelvar, levels = nis_var)
test_nis$Modelvar = factor(test_nis$Modelvar, levels = nis_var)

summary(train_nis$Modelvar)
summary(test_nis$Modelvar)

# standardize Model levels
nis_mdl = unique(c(levels(train_nis$Model),levels(test_nis$Model)))
train_nis$Model = factor(train_nis$Model, levels = nis_mdl)
test_nis$Model = factor(test_nis$Model, levels = nis_mdl)

summary(train_nis$Model)
summary(test_nis$Model)

# standardize MfgYr levels
nis_yr = unique(c(levels(train_nis$MfgYr), levels(test_nis$MfgYr)))
train_nis$MfgYr = factor(train_nis$MfgYr, levels = nis_yr)
test_nis$MfgYr = factor(test_nis$MfgYr, levels = nis_yr)

summary(train_nis$MfgYr)
summary(test_nis$MfgYr)

# Nissan: identify predictors ---------------------------------------------
to_drop = as.character()
for (i in 9:(length(colnames(train_nis))-2)){ # ugly syntax: did not rearrange columns!!
  test = plyr::count(train_nis, colnames(train_nis)[i])
  if (length(test$freq) == 1){ 
    # alternative we can add: if length(test$freq) > 1, then test$freq[1]/test$freq[2] < 5% then remove
    to_drop = c(paste(colnames(train_nis)[i]), to_drop)
  }
}

drops = c(to_drop, "ID", "Brand", "Features", "Price", "CC", "CC_adj")

# drop all non-relevant predictors
train_nis = train_nis[, !names(train_nis) %in% drops]
test_nis = test_nis[, !names(test_nis) %in% drops]

colnames(train_nis) == colnames(test_nis)
colnames(train_nis)
str(train_nis)

# Nissan: SVM -------------------------------------------------------------

system.time(tune_radial_nis <- tune(svm, Modelvar ~ ., data = train_nis,
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

dim(train_nis)

# 7,848 obs, 11 var
# user  system elapsed 
# 2728.99    9.68 2820.52 ~47 min

# 3,407 obs, 19-5 variables
# user  system elapsed 
# 595.82    0.61  613.63 ~10 min

svm_nis = svm(Modelvar ~., data = train_nis,
              kernel="radial", cost=tune_radial_nis$best.parameters$cost,
              gamma=tune_radial_nis$best.parameters$gamma, trace=F
)

svm_est_nis = predict(svm_nis, newdata=test_nis)

nis_est_lvl = unique(c(levels(svm_est_nis), levels(test_nis$Modelvar)))
test_nis$Modelvar = factor(test_nis$Modelvar, levels = nis_est_lvl)
svm_est_nis = factor(svm_est_nis, levels = nis_est_lvl)

1-sum(test_nis$Modelvar==svm_est_nis)/nrow(test_nis)

nis_tbl = table(test_nis$Modelvar, svm_est_nis)
write.csv(nis_tbl,"nis.csv")

c(colnames(train_nis), dim(train_nis))

# Honda: All --------------------------------------------------------------

train_hon = dplyr::filter(training_data, Brand=="HONDA")
test_hon = dplyr::filter(testing_data, Brand=="HONDA")

train_hon$CC_scl = as.numeric(scale(train_hon$CC_adj))
test_hon$CC_scl = as.numeric(scale(test_hon$CC_adj))

# filter training set only for variants with size >20
train_hon = train_hon %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>20) %>%
  as.data.frame()

train_hon = droplevels(train_hon)
test_hon = droplevels(test_hon)

# standardize Variant levels
hon_var = unique(c(levels(train_hon$Modelvar),levels(test_hon$Modelvar)))
train_hon$Modelvar = factor(train_hon$Modelvar, levels = hon_var)
test_hon$Modelvar = factor(test_hon$Modelvar, levels = hon_var)

summary(train_hon$Modelvar)
summary(test_hon$Modelvar)

# standardize Model levels
hon_mdl = unique(c(levels(train_hon$Model),levels(test_hon$Model)))
train_hon$Model = factor(train_hon$Model, levels = hon_mdl)
test_hon$Model = factor(test_hon$Model, levels = hon_mdl)

summary(train_hon$Model)
summary(test_hon$Model)

# standardize MfgYr levels
hon_yr = unique(c(levels(train_hon$MfgYr), levels(test_hon$MfgYr)))
train_hon$MfgYr = factor(train_hon$MfgYr, levels = hon_yr)
test_hon$MfgYr = factor(test_hon$MfgYr, levels = hon_yr)

summary(train_hon$MfgYr)
summary(test_hon$MfgYr)

# Honda: identify predictors ----------------------------------------------
to_drop = as.character()
for (i in 9:(length(colnames(train_hon))-2)){
  test = plyr::count(train_hon, colnames(train_hon)[i])
  if (length(test$freq) == 1){ 
    # alternative we can add: if length(test$freq) > 1, then test$freq[1]/test$freq[2] < 5% then remove
    to_drop = c(paste(colnames(train_hon)[i]), to_drop)
  }
}

drops = c(to_drop, "ID", "Brand", "Features", "Price", "CC", "CC_adj")

# drop all non-relevant predictors
train_hon = train_hon[, !names(train_hon) %in% drops]
test_hon = test_hon[, !names(test_hon) %in% drops]

colnames(train_hon) == colnames(test_hon)
colnames(train_hon)

# Honda: SVM --------------------------------------------------------------

system.time(tune_radial_hon <- tune(svm, Modelvar ~ ., data = train_hon,
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

dim(train_hon)

# 15k obs, 10 var
# user  system elapsed 
# 9147.26   25.03 9379.13 ~156 mins

# 7,848 obs, 11 var
# user  system elapsed 
# 2728.99    9.68 2820.52 ~47 min

# 3,407 obs, 19-5 variables
# user  system elapsed 
# 595.82    0.61  613.63 ~10 min

svm_hon = svm(Modelvar ~., data = train_hon,
              kernel="radial", cost=tune_radial_hon$best.parameters$cost,
              gamma=tune_radial_hon$best.parameters$gamma, trace=F
)

svm_est_hon = predict(svm_hon, newdata=test_hon)

hon_est_lvl = unique(c(levels(svm_est_hon), levels(test_hon$Modelvar)))
test_hon$Modelvar = factor(test_hon$Modelvar, levels = hon_est_lvl)
svm_est_hon = factor(svm_est_hon, levels = hon_est_lvl)

1-sum(test_hon$Modelvar==svm_est_hon)/nrow(test_hon)

hon_tbl = table(test_hon$Modelvar, svm_est_hon)
write.csv(hon_tbl,"hon.csv")

c(colnames(train_hon), dim(train_hon))

# Toyota: All -------------------------------------------------------------
train_tyt = dplyr::filter(training_data, Brand=="Toyota")
head(plyr::count(train_tyt$Modelvar))

test_tyt = dplyr::filter(testing_data, Brand=="Toyota" & 
                               !Modelvar %in% c('', '-', '0', '#N/A'))
head(plyr::count(test_tyt$Modelvar))

# clean up spaces / upper-lower char and re-factor
train_tyt$Modelvar = as.factor(toupper(trimws(train_tyt$Modelvar)))
test_tyt$Modelvar = as.factor(toupper(trimws(test_tyt$Modelvar)))

plyr::count(train_tyt$Modelvar)
plyr::count(test_tyt$Modelvar)

plyr::count(train_tyt$CC_adj)
plyr::count(test_tyt$CC_adj)

train_tyt$CC_scl = as.numeric(scale(train_tyt$CC_adj))
test_tyt$CC_scl = as.numeric(scale(test_tyt$CC_adj))

# filter for variants with size > 30
train_tyt = train_tyt %>% 
  group_by(Modelvar) %>%
  filter(n()>30) %>%
  as.data.frame()

test_tyt = test_tyt %>% 
  group_by(Modelvar) %>%
  filter(n()>30) %>%
  as.data.frame()

train_tyt = droplevels(train_tyt)
test_tyt = droplevels(test_tyt)

# standardize Variant levels
tyt_var = unique(c(levels(train_tyt$Modelvar),levels(test_tyt$Modelvar)))
train_tyt$Modelvar = factor(train_tyt$Modelvar, levels = tyt_var)
test_tyt$Modelvar = factor(test_tyt$Modelvar, levels = tyt_var)

# standardize Model levels
tyt_mdl = unique(levels(train_tyt$Model),levels(test_tyt$Model))
train_tyt$Model = factor(train_tyt$Model, levels = tyt_mdl)
test_tyt$Model = factor(test_tyt$Model, levels = tyt_mdl)

# standardize MfgYr levels
tyt_yr = unique(levels(train_tyt$MfgYr), levels(test_tyt$MfgYr))
train_tyt$MfgYr = factor(train_tyt$MfgYr, levels = tyt_yr)
test_tyt$MfgYr = factor(test_tyt$MfgYr, levels = tyt_yr)


summary(train_tyt) # check for NAs
summary(test_tyt) # check for NAs


# Toyota: identify predictors ---------------------------------------------

plyr::count(train_tyt, "Transm")
plyr::count(train_tyt, "CC")
plyr::count(train_tyt, "Airbag")
plyr::count(train_tyt, "Leather")
plyr::count(train_tyt, "Nav")
plyr::count(train_tyt, "ABS")
plyr::count(train_tyt, "SportRims")
plyr::count(train_tyt, "RevCam")
plyr::count(train_tyt, "PowDoor") # weird some units have power door?
plyr::count(train_tyt, "TouchScreen") # all N/A
plyr::count(train_tyt, "ClimaCtrl")

test_search = grepl("leather", test_tyt$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

test_search = grepl("light", test_tyt$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

# drop non-useful predictors
drops = c("CC", "TouchScreen")
train_tyt = train_tyt[, !names(train_tyt) %in% drops]
test_tyt = test_tyt[, !names(test_tyt) %in% drops]

leather_tytTr = grepl("leather", train_tyt$Features, ignore.case=TRUE)
leather_tytTt = grepl("leather", test_tyt$Features, ignore.case=TRUE)

light_tytTr = grepl("light", train_tyt$Features, ignore.case=TRUE)
light_tytTt = grepl("light", test_tyt$Features, ignore.case=TRUE)

train_tyt$Leather = leather_tytTr
test_tyt$Leather = leather_tytTt

train_tyt$Light = light_tytTr
test_tyt$Light = light_tytTt

colnames(train_tyt) == colnames(test_tyt)

colnames(train_tyt)
str(train_tyt)

# SVM: Toyota -------------------------------------------------------------

library(e1071)
system.time(tune_radial_tyt <- tune(svm, Modelvar ~ ., data = train_tyt[,-c(1:2, 4:5, 17)],
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 11,400 obs, 19-5 variables
# user  system elapsed 
# 6086.24   18.04 6198.82 ~103

# 11,400 obs, 17-5 variables
# user  system elapsed 
# 5827.84   15.63 6342.66 ~105

tune_radial_tyt$best.parameters

# excl. manf yr, CC cost=10, gamma=0.1

svm_tyt = svm(Modelvar ~., data = train_tyt[,-c(1:2, 4:5, 17)],
             kernel="radial", cost=tune_radial_tyt$best.parameters$cost,
             gamma=tune_radial_tyt$best.parameters$gamma, trace=F
)



svm_est_tyt = predict(svm_tyt, newdata=test_tyt[,-c(1:2, 4:5, 17)])

1-sum(test_tyt[,6]==svm_est_tyt)/nrow(test_tyt)

# Error info:
# 17.4% (incl MfgYr, CC)
# 30.4% (excl MfgYr, CC)

tyt_tbl = table(test_tyt[,6], svm_est_tyt)
write.csv(tyt_tbl,"tyt.csv")


# Perodua: All ------------------------------------------------------------
train_perodua = dplyr::filter(training_data, Brand=="Perodua")
head(plyr::count(train_perodua$Modelvar))

test_perodua = dplyr::filter(testing_data, Brand=="Perodua" & 
                              !Modelvar %in% c('', '-', '0', '#N/A'))
head(plyr::count(test_perodua$Modelvar))

train_perodua$Modelvar = as.character(train_perodua$Modelvar)
test_perodua$Modelvar = as.character(test_perodua$Modelvar)

# clean up spaces / upper-lower char and re-factor
train_perodua$Modelvar = as.factor(toupper(trimws(train_perodua$Modelvar)))
test_perodua$Modelvar = as.factor(toupper(trimws(test_perodua$Modelvar)))

plyr::count(train_perodua$Modelvar)
plyr::count(test_perodua$Modelvar)

# filter for variants with size > 30
train_perodua = train_perodua %>% 
  group_by(Modelvar) %>%
  filter(n()>30) %>%
  as.data.frame()

test_perodua = test_perodua %>% 
  group_by(Modelvar) %>%
  filter(n()>30) %>%
  as.data.frame()

train_perodua = droplevels(train_perodua)
test_perodua = droplevels(test_perodua)

p2_lvl = unique(c(levels(train_perodua$Modelvar),levels(test_perodua$Modelvar)))
train_perodua$Modelvar = factor(train_perodua$Modelvar, levels = p2_lvl)
test_perodua$Modelvar = factor(test_perodua$Modelvar, levels = p2_lvl)

plyr::count(train_perodua$CC_adj)
train_perodua = dplyr::filter(train_perodua, CC_adj <= 1600) # alternative filter out data with low obs
train_perodua$CC_scl = scale(train_perodua$CC_adj)

plyr::count(test_perodua$Modelvar)
test_perodua = dplyr::filter(test_perodua, CC_adj <= 1600) # alternative filter out data with low obs
test_perodua$CC_scl = scale(test_perodua$CC_adj)
test_perodua$CC_scl = as.numeric(test_perodua$CC_scl)

summary(train_perodua) # check for NAs
summary(test_perodua) # check for NAs

# P2: Identify predictors-----------------------------------------------------

plyr::count(train_perodua, "Transm") # filtered for autos
plyr::count(train_perodua, "CC") # 3 different ways of CC Groups: 12-20, 1000-2000, >10000
plyr::count(train_perodua, "Airbag")
plyr::count(train_perodua, "Leather")
plyr::count(train_perodua, "Nav")
plyr::count(train_perodua, "ABS")
plyr::count(train_perodua, "SportRims")
plyr::count(train_perodua, "RevCam")
plyr::count(train_perodua, "PowDoor") # weird some units have power door?
plyr::count(train_perodua, "TouchScreen") # all N/A
plyr::count(train_perodua, "ClimaCtrl")

test_search = grepl("leather", test_perodua$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

test_search = grepl("light", test_perodua$Features, ignore.case=TRUE)
plyr::count(test_search) # OK


# drop non-useful predictors
drops = c("CC", "TouchScreen")
train_perodua = train_perodua[, !names(train_perodua) %in% drops]
test_perodua = test_perodua[, !names(test_perodua) %in% drops]

# add extra predictors
leather_p2Tr = grepl("leather", train_perodua$Features, ignore.case=TRUE)
leather_p2Tt = grepl("leather", test_perodua$Features, ignore.case=TRUE)

light_p2Tr = grepl("light", train_perodua$Features, ignore.case=TRUE)
light_p2Tt = grepl("light", test_perodua$Features, ignore.case=TRUE)

train_perodua$Leather = leather_p2Tr
test_perodua$Leather = leather_p2Tt

train_perodua$Light = light_p2Tr
test_perodua$Light = light_p2Tt

colnames(train_perodua) == colnames(test_perodua)

colnames(train_perodua)

str(train_perodua)


# P2: SVM -----------------------------------------------------------------

library(e1071)
system.time(svm_tune_radial <- tune(svm, Modelvar ~ ., data = train_perodua[,-c(1:2, 4:5, 17)],
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 9400 obs, 19-5 variables
# user  system elapsed 
# 3655.37    7.22 3680.47  ~61 mins

# 9400 obs, 17-5 variables
# user  system elapsed 
# 3140.10    8.92 3261.57 ~54 mins

svm_tune_radial$best.parameters

# excl. manf yr, CC cost=10, gamma=0.1

svm_p2 = svm(Modelvar ~., data = train_perodua[,-c(1:2, 4:5, 17)],
                 kernel="radial", cost=svm_tune_radial$best.parameters$cost,
                 gamma=svm_tune_radial$best.parameters$gamma, trace=F
)


svm_p2_est = predict(svm_p2, newdata=test_perodua[,-c(1:2, 4:5, 17)])

1-sum(test_perodua[,6]==svm_p2_est)/nrow(test_perodua)

# Error info:
# 20% (incl MfgYr, CC)
# 37% (excl MfgYr, CC)

p2_tbl = table(test_perodua[,6], svm_p2_est)
write.csv(p2_tbl,"p2.csv")


# Proton: All -------------------------------------------------------------

# Note: Check for NAs in data first
train_proton = dplyr::filter(training_data, Brand=="Proton")
plyr::count(train_proton$Modelvar)

test_proton = dplyr::filter(testing_data, Brand=="Proton" & 
                              !Modelvar %in% c('', '-', '0', '#N/A'))
head(plyr::count(test_proton$Modelvar))

train_proton$Modelvar = as.character(train_proton$Modelvar)
test_proton$Modelvar = as.character(test_proton$Modelvar)

# clean up spaces / upper-lower char and re-factor
train_proton$Modelvar = as.factor(toupper(trimws(train_proton$Modelvar)))
test_proton$Modelvar = as.factor(toupper(trimws(test_proton$Modelvar)))

plyr::count(train_proton$Modelvar)
plyr::count(test_proton$Modelvar)

# filter for variants with size > 30
train_proton = train_proton %>% 
  group_by(Modelvar) %>%
  filter(n()>30) %>%
  as.data.frame()

test_proton = test_proton %>% 
  group_by(Modelvar) %>%
  filter(n()>30) %>%
  as.data.frame()

train_proton = droplevels(train_proton)
test_proton = droplevels(test_proton)

proton_lvl = unique(c(levels(train_proton$Modelvar),levels(test_proton$Modelvar)))
train_proton$Modelvar = factor(train_proton$Modelvar, levels = proton_lvl)
test_proton$Modelvar = factor(test_proton$Modelvar, levels = proton_lvl)

all.equal.factor(train_proton$Modelvar, test_proton$Modelvar)

# standardize models
proton_mdl = unique(levels(train_proton$Model), levels(test_proton$Model))
train_proton$Model = factor(train_proton$Model, levels = proton_mdl)
test_proton$Model = factor(test_proton$Model, levels = proton_mdl)

train_proton$CC_scl = scale(train_proton$CC_adj)
test_proton$CC_scl = scale(test_proton$CC_adj)


# Identify predictors: Proton All ---------------------------------------------------------

# Identify useful predictors
plyr::count(train_proton, "Transm") # filtered for autos
plyr::count(train_proton, "CC") # 3 different ways of CC Groups: 12-20, 1000-2000, >10000
plyr::count(train_proton, "Airbag")
plyr::count(train_proton, "Leather")
plyr::count(train_proton, "Nav")
plyr::count(train_proton, "ABS")
plyr::count(train_proton, "SportRims")
plyr::count(train_proton, "RevCam")
plyr::count(train_proton, "PowDoor")
plyr::count(train_proton, "TouchScreen") # all N/A
plyr::count(train_proton, "ClimaCtrl")

test_search = grepl("leather", test_proton$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

test_search = grepl("light", test_proton$Features, ignore.case=TRUE)
plyr::count(test_search) # OK

# drop non-useful predictors
drops = c("CC", "TouchScreen")
train_proton = train_proton[, !names(train_proton) %in% drops]
test_proton = test_proton[, !names(test_proton) %in% drops]

leather_protonTr = grepl("leather", train_proton$Features, ignore.case=TRUE)
leather_protonTt = grepl("leather", test_proton$Features, ignore.case=TRUE)

light_protonTr = grepl("light", train_proton$Features, ignore.case=TRUE)
light_protonTt = grepl("light", test_proton$Features, ignore.case=TRUE)

train_proton$Leather = leather_protonTr
test_proton$Leather = leather_protonTt

train_proton$Light = light_protonTr
test_proton$Light = light_protonTt

colnames(train_proton) == colnames(test_proton)

colnames(train_proton)
str(train_proton)

# SVM: Proton All ---------------------------------------------------------



colnames(train_proton)
library(e1071)
system.time(svm_tune_radial <- tune(svm, Modelvar ~ ., data = train_proton[,-c(1:2, 4:5, 17)],
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

# 10,000 obs, (19-5) var
# user  system elapsed 
# 8450.45   21.45 8853.12 ~147 mins

# 10,000 obs, (17-5) var
# user  system elapsed 
# 5458.87   17.33 6132.82 ~102 mins
svm_tune_radial$best.parameters 

# excl. manf yr, CC cost=10, gamma=0.1 

svm_proton = svm(Modelvar ~., data = train_proton[,-c(1:2, 4:5, 17)],
                 kernel="radial", cost=svm_tune_radial$best.parameters$cost,
                 gamma=svm_tune_radial$best.parameters$gamma, trace=F
)


sum(is.na(test_proton)) # 963 NAs
summary(test_proton) # 963 NAs located at Model
str(test_proton)

test_proton$CC_scl = as.numeric(test_proton$CC_scl) # convert list to 1d vectors
test_proton_f = dplyr::filter(test_proton, !is.na(Model))
summary(test_proton_f)

svm_proton_est = predict(svm_proton, newdata=test_proton_f[,-c(1:2, 4:5, 17)])

1-sum(test_proton_f[,6]==svm_proton_est)/nrow(test_proton_f)

# Error info:
# 16% (incl MfgYr, CC)
# 32% (took out NAs in Model, excl MfgYr, CC)

proton_tbl = table(test_proton_f[,6], svm_proton_est)
write.csv(proton_tbl,"proton.csv")


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
