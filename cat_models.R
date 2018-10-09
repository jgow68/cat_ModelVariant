

# Read data from csv ------------------------------------------------------

setwd("~/Github Folder/cat_ModelVariant")

# form training df
data1 = read.csv('Sept18_V2.csv')
data2 = read.csv('Aug18_V3.csv')

# starting from "Aug'18 dataset has vehicle variant and series obtained from Mudah
# issue is those with variant will not provide features info, hence we try to use seller comments instead (text extract)
# can use provided variant/series as predictor, but unsure how accurate this is going forward

# data1 = read.csv('May18_V2.csv')
# data2 = read.csv('Apr18_V2.csv')
# data3 = read.csv('Mar18_V1.csv')

# data1 = read.csv("Feb18_V1.csv")
# data2 = read.csv('Jan18_V1.csv')
# data3 = read.csv('Dec17_V1.csv')
# data4 = read.csv('Nov17_V3.csv')

# form testing df

verify_data = data1
# verify_data = read.csv("Jun18_V2.csv")
# verify_data = read.csv("Mar18_V1.csv")


# substitute features with seller comments
data2$Features = as.character(data2$Features)
data2$Seller_Comments = as.character(data2$Seller_Comments)

data2$Text = ifelse(data2$Variant == "", 
                    data2$Features, data2$Seller_Comments)

data2$Transmission = as.factor(
  ifelse(grepl("manual", data2$Transmission, ignore.case = TRUE), 
         "MANUAL", "AUTO"))

# search text if certain top features appeared
library(quanteda)
quanteda_options("threads" = 4)
corpus_text = corpus(as.character(data2$Text)) # creates corpus

dfm_text <- dfm( # creates document feature matrix
  corpus_text, ngrams = 1,
  remove = stopwords("english"), remove_punct = TRUE, remove_numbers = TRUE, stem = FALSE
)

tf_text <- topfeatures(dfm_text, n = 50, decreasing=TRUE)
tf_text

# search Features for key words and form a new column to merge with dataframe

data1$Features = as.character(data1$Features)
data2$Features = as.character(data2$Features)
data3$Features = as.character(data3$Features)
# data4$Features = as.character(data4$Features)
verify_data$Features = as.character(verify_data$Features)

# New Data 2 Search (combined Features/Seller Comments)
data2$Airbag = grepl("Airbag", data2$Text)
data2$Leather = grepl("Leather", data2$Text)
data2$Nav = grepl("Nav", data2$Text)
data2$ABS = grepl("ABS", data2$Text)
data2$Sport.Rim = grepl("Sport rim", data2$Text)
data2$Reverse.Camera = grepl("Reverse camera", data2$Text)
data2$Power.Door = grepl("Power Door", data2$Text)
data2$Climate.Control = grepl("Climate control", data2$Text)
data2$Light = grepl("Light", data2$Text)

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


# Decide dataset to merge -------------------------------------------------

# 1st case: choose most recent data set for training
training_data = training_data2

# 2nd case: merge historical datasets by outer join 2 df
# consider using data.table for faster computation

# # Remove duplicates that exist in data set right from the start
# 
# training_data1 = training_data1[!duplicated(training_data1$ID), ]
# training_data2 = training_data2[!duplicated(training_data2$ID), ]
# training_data3 = training_data3[!duplicated(training_data3$ID), ]
# 
# # training_data_merged = merge(training_data1, training_data2, all=TRUE) # use left join to prioritize newer datasets
# # training_data = merge(training_data, training_data3, all=TRUE)
# # training_data = merge(training_data, training_data4, all=TRUE)
# 
# rm(training_data) # reset training data
# 
# # Merge data set for training (Can be improved)
# 
# training_data_merged = dplyr::left_join(training_data1, training_data2)
# 
# head(training_data_merged[duplicated(training_data_merged$ID), ])
# 
# ID_init = training_data_merged$ID
# training_data_init = dplyr::filter(training_data2, !(ID %in% ID_init))
# 
# training_data = merge(training_data_merged, training_data_init, all=TRUE)
# 
# head(training_data[duplicated(training_data$ID), ])
# 
# training_data = merge(training_data, training_data2[!ID_init, ], all=TRUE)
# 
# head(training_data[duplicated(training_data$ID), ])
# 
# # now merge the 3rd dataset
# ID_init = training_data$ID
# training_data_init = dplyr::filter(training_data3, !(ID %in% ID_init))
# 
# training_data = merge(training_data, training_data_init, all=TRUE)
# head(training_data[duplicated(training_data$ID), ])

dim(training_data)
# merge Mar-May'18 ~210k obs


# Prep merged data ---------------------------------------------------------------

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

# Testing for new dataset format (+Variant/Series) ------------------------

train_all = training_data

# filter training set only for variants with size >20
train_all = train_all %>% 
  dplyr::group_by(Modelvar) %>%
  dplyr::filter(n()>20) %>%
  as.data.frame()

train_all = droplevels(train_all)

colnames(train_all)
to_drop = as.character()
for (i in 9:(length(colnames(train_all)))){ # check if the loop range is correct
  test = plyr::count(train_all, colnames(train_all)[i])
  if (length(test$freq) == 1){ 
    # alternative we can add: if length(test$freq) > 1, then test$freq[1]/test$freq[2] < 5% then remove
    to_drop = c(paste(colnames(train_all)[i]), to_drop)
  }
}

drops = c(to_drop, "ID", "Features", "Price", "CC")

# drop all non-relevant predictors
train_all = train_all[, !names(train_all) %in% drops]

str(train_all)


# Create train, valid, test set with caret ------------------------------------

library(caret)
set.seed(123)
trainIndex <- createDataPartition(train_all$Modelvar, p = .7, 
                                  list = FALSE, 
                                  times = 1)

df_split_train = train_all[trainIndex, ]
df_split_nonTrain = train_all[-trainIndex, ]

validIndex <- createDataPartition(df_split_nonTrain$Modelvar, p = .5, 
                                  list = FALSE, 
                                  times = 1)

df_split_valid = df_split_nonTrain[validIndex, ]
df_split_test = df_split_nonTrain[-validIndex, ]

dim(df_split_train)
dim(df_split_valid)
dim(df_split_test)

# h2o glm -----------------------------------------------------------------

h2o.init(min_mem_size="4g", max_mem_size = "8g")

df_train <- as.h2o(df_split_train)
df_valid <- as.h2o(df_split_valid)
df_test <- as.h2o(df_split_test)

y <- 'Modelvar'
x <- setdiff(names(df_train), y)

system.time(glm_fit1 <- h2o.glm(x = x, 
                    y = y, 
                    training_frame = df_train,
                    validation_frame = df_valid,
                    model_id = "glm_fit1",
                    family = "multinomial"))

system.time(glm_fit2 <- h2o.glm(x = x, 
                                y = y, 
                                training_frame = df_train,
                                model_id = "glm_fit2",
                                validation_frame = df_valid,
                                family = "multinomial",
                                lambda_search = TRUE))

# Let's compare the performance of the two GLMs
glm_perf1 <- h2o.performance(model = glm_fit1,
                             newdata = df_test)
glm_perf2 <- h2o.performance(model = glm_fit2,
                             newdata = df_test)


h2o.mse(glm_perf1) # 0.2378
h2o.mse(glm_perf2) # 0.2176

# AutoML (too long to run for full dataset)------------------------------------------------------------------

library(h2o)
h2o.init(min_mem_size="4g", max_mem_size = "8g")
h2o.shutdown()

df_train <- as.h2o(df_split_train)
df_valid <- as.h2o(df_split_valid)
df_test <- as.h2o(df_split_test)

y <- 'Modelvar'
x <- setdiff(names(df_train), y)

AML_all <- h2o.automl(x = x,
                     y = y,
                     training_frame = df_train,
                     nfolds = 5,
                     keep_cross_validation_predictions = TRUE,
                     validation_frame = df_valid,
                     leaderboard_frame = df_test,
                     exclude_algos = c("DeepLearning"), # exclude_algos = c("GLM", "DeepLearning", "GBM", DRF", "StackedEnsemble"),
                     #max_runtime_secs = 60, 
                     max_models = 1,
                     seed = 1,
                     project_name = "Class_all_Aug_Sept18"
)

print(AML_all@leaderboard)

h2o.mean_per_class_error(AML_all@leader)
h2o.performance(AML_all@leader)


h2o.getModel(AML_all@leader)


AML_all@leader

# Perodua: All ------------------------------------------------------------

train_p2 = dplyr::filter(training_data, Brand=="PERODUA")
test_p2 = dplyr::filter(testing_data, Brand=="PERODUA")

train_p2$CC_scl = as.numeric(scale(train_p2$CC_adj))
test_p2$CC_scl = as.numeric(scale(test_p2$CC_adj))

# filter training set only for variants with size >20
train_p2 = train_p2 %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>20) %>%
  as.data.frame()

train_p2 = droplevels(train_p2)
test_p2 = droplevels(test_p2)

# standardize Variant levels
p2_var = unique(c(levels(train_p2$Modelvar),levels(test_p2$Modelvar)))
train_p2$Modelvar = factor(train_p2$Modelvar, levels = p2_var)
test_p2$Modelvar = factor(test_p2$Modelvar, levels = p2_var)

summary(train_p2$Modelvar)
summary(test_p2$Modelvar)

# standardize Model levels
p2_mdl = unique(c(levels(train_p2$Model),levels(test_p2$Model)))
train_p2$Model = factor(train_p2$Model, levels = p2_mdl)
test_p2$Model = factor(test_p2$Model, levels = p2_mdl)

summary(train_p2$Model)
summary(test_p2$Model)

# standardize MfgYr levels
p2_yr = unique(c(levels(train_p2$MfgYr), levels(test_p2$MfgYr)))
train_p2$MfgYr = factor(train_p2$MfgYr, levels = p2_yr)
test_p2$MfgYr = factor(test_p2$MfgYr, levels = p2_yr)

summary(train_p2$MfgYr)
summary(test_p2$MfgYr)

str(train_p2)
str(test_p2)

# P2: Identify predictors-----------------------------------------------------
to_drop = as.character()
for (i in 9:(length(colnames(train_p2))-2)){
  test = plyr::count(train_p2, colnames(train_p2)[i])
  if (length(test$freq) == 1){ 
    # alternative we can add: if length(test$freq) > 1, then test$freq[1]/test$freq[2] < 5% then remove
    to_drop = c(paste(colnames(train_p2)[i]), to_drop)
  }
}

drops = c(to_drop, "ID", "Brand", "Features", "Price", "CC", "CC_adj")

# drop all non-relevant predictors
train_p2 = train_p2[, !names(train_p2) %in% drops]
test_p2 = test_p2[, !names(test_p2) %in% drops]

colnames(train_p2) == colnames(test_p2)
colnames(train_p2)

# P2: SVM -----------------------------------------------------------------
dim(train_p2)

system.time(tune_radial_p2 <- tune(svm, Modelvar ~ ., data = train_p2,
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)



svm_p2 = svm(Modelvar ~., data = train_p2,
              kernel="radial", cost=tune_radial_p2$best.parameters$cost,
              gamma=tune_radial_p2$best.parameters$gamma, trace=F
)

svm_est_p2 = predict(svm_p2, newdata=test_p2)
plyr::count(svm_est_p2)
p2_est_lvl = unique(c(levels(svm_est_p2), levels(test_p2$Modelvar)))
test_p2$Modelvar = factor(test_p2$Modelvar, levels = p2_est_lvl)
svm_est_p2 = factor(svm_est_p2, levels = p2_est_lvl)

1-sum(test_p2$Modelvar==svm_est_p2)/nrow(test_p2)

p2_tbl = table(test_p2$Modelvar, svm_est_p2)
write.csv(p2_tbl,"p2.csv")

c(colnames(train_p2), dim(train_p2))

# 18k obs, 11 var
# user   system  elapsed 
# 12621.70    30.49 12797.28 ~ 213 mins, 3.5 hrs

# 9400 obs, 19-5 variables
# user  system elapsed 
# 3655.37    7.22 3680.47  ~61 mins

# 9400 obs, 17-5 variables
# user  system elapsed 
# 3140.10    8.92 3261.57 ~54 mins




# P2: h2o GLM -------------------------------------------------------------

h2o.init(nthreads = -1, 
         min_mem_size="4g", max_mem_size = "8g"
         )

data = as.h2o(train_p2)
data_for_pred = as.h2o(test_p2[, -2]) # removed the variant from the predidction set, if not prediction will trigger errors

y <- 'Modelvar'
x <- setdiff(names(data), y)

splits <- h2o.splitFrame(data = data, 
                         ratios = c(0.7, 0.15),  #partition data into 70%, 15%, 15% chunks
                         seed = 1)  #setting a seed will guarantee reproducibility
train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

glm_fit1 <- h2o.glm(x = x, 
                    y = y, 
                    training_frame = train,
                    model_id = "glm_fit1",
                    family = "multinomial"
)  #similar to R's glm, h2o.glm has the family argument
# binomial, gaussian (int/num), ordinal (cat>3 lvls), quasibinomial (num), poisson, gamma, tweedie

# automatic tuning of lambda, for regularization
system.time(glm_fit2 <- h2o.glm(x = x, 
                        y = y, 
                        training_frame = train,
                        model_id = "glm_fit2",
                        validation_frame = valid,
                        family = "multinomial",
                        lambda_search = TRUE)
) # 981.54s ~ 15 mins for full data # 684s ~ 11 mins for split data: train + valid

# Let's compare the performance of the two GLMs
glm_perf1 <- h2o.performance(model = glm_fit1,
                             newdata = test)
glm_perf2 <- h2o.performance(model = glm_fit2,
                             newdata = test)

# Print model performance
# glm_perf1
# glm_perf2

h2o.mse(glm_perf1) # 0.2378
h2o.mse(glm_perf2) # 0.2176

# AUC applicable for binomial classification
# h2o.auc(glm_perf1)  
# h2o.auc(glm_perf2)

# Error: DistributedException: 'null', caused by java.lang.ArrayIndexOutOfBoundsException
# error due to unarranged x, y for test set


glm_pred_p2 = h2o.predict(glm_fit2, newdata = data_for_pred)
p2_output_glm = as.data.frame(test_p2$Modelvar)
p2_output_glm$est_var = glm_pred_p2[, 1] # first column is the predictions
# p2_output_glm$est_var = colnames(glm_pred_p2[, -1])[max.col(glm_pred_p2[, -1], ties.method="first")]
colnames(p2_output_glm) = c('act_var', 'est_var')

dim(plyr::count(p2_output_glm$est_var)) # identified no. of variants predicted

sum(p2_output_glm$est_var == p2_output_glm$act_var)/nrow(p2_output_glm) # accuracy level

p2_glm_for_output = as.data.frame(p2_output_glm) # weird format, blanks repl with 'X'

# write.csv(p2_glm_for_output, file = 'glm_p2.csv')

# P2: h2o Random Forest ---------------------------------------------------

# Ref: https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/intro-to-h2o.R

h2o.init(nthreads = -1, 
         min_mem_size="4g", max_mem_size = "8g"
)


rf_fit1 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit1",
                            seed = 1)


rf_fit2 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit2",
                            #validation_frame = valid,  #only used if stopping_rounds > 0
                            ntrees = 100, # default 50, more shud increase perf
                            seed = 1)

# Let's compare the performance of the two RFs
rf_perf1 <- h2o.performance(model = rf_fit1,
                            newdata = test)
rf_perf2 <- h2o.performance(model = rf_fit2,
                            newdata = test)

# Print model performance
rf_perf1
rf_perf2

# Retreive test set AUC
h2o.mse(rf_perf1) # 0.2111
h2o.mse(rf_perf2) # 0.2108

system.time(rf_fit3 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit3",
                            seed = 1,
                            nfolds = 5) # cross-validation perf
) # 92.68s, 1.5 mins

# h2o.mse(rf_fit3, xval=TRUE)

rf_pred_p2 = h2o.predict(rf_fit3, data_for_pred) # need to manually id class with highest prob per row

p2_output_rf = as.data.frame(test_p2$Modelvar)
p2_output_rf$est_var = rf_pred_p2[, 1] # colnames(rf_pred_p2[, -1])[max.col(rf_pred_p2[, -1], ties.method="first")]
colnames(p2_output_rf) = c('act_var', 'est_var')

dim(plyr::count(p2_output_rf$est_var)) # identified no. of variants predicted
sum(p2_output_rf$est_var == p2_output_rf$act_var)/nrow(p2_output_rf) # accuracy level

# P2: h2o GBM -------------------------------------------------------------

gbm_fit1 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit1",
                    seed = 1)

gbm_fit2 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit2",
                    #validation_frame = valid,  #only used if stopping_rounds > 0
                    ntrees = 500, # default 50, need to take care of overfitting
                    seed = 1)

gbm_fit3 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit3",
                    validation_frame = valid,  #only used if stopping_rounds > 0
                    ntrees = 500,
                    score_tree_interval = 5,      #used for early stopping
                    stopping_rounds = 3,          #used for early stopping
                    stopping_metric = "AUTO",      #used for early stopping
                    stopping_tolerance = 0.0005,  #used for early stopping
                    seed = 1)

system.time(gbm_fit4 <- h2o.gbm(
                        ## standard model parameters
                        x = x, 
                        y = y, 
                        training_frame = train, 
                        validation_frame = valid,
                        model_id = "gbm_fit4",
                        
                        ## more trees is better if the learning rate is small enough 
                        ## here, use "more than enough" trees - we have early stopping
                        ntrees = 10000,                                                            
                        
                        ## smaller learning rate is better (this is a good value for most datasets, but see below for annealing)
                        learn_rate=0.01,                                                         
                        
                        ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
                        stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUTO", 
                        
                        ## sample 80% of rows per tree
                        sample_rate = 0.8,                                                       
                        
                        ## sample 80% of columns per split
                        col_sample_rate = 0.8,                                                   
                        
                        ## fix a random number generator seed for reproducibility
                        seed = 1,                                                             
                        
                        ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
                        score_tree_interval = 10                                                 
                      )
) # 118.95s, 2 mins

gbm_perf1 <- h2o.performance(model = gbm_fit1,
                             newdata = test)
gbm_perf2 <- h2o.performance(model = gbm_fit2,
                             newdata = test)
gbm_perf3 <- h2o.performance(model = gbm_fit3,
                             newdata = test)
gbm_perf4 <- h2o.performance(model = gbm_fit4,
                             newdata = test)

# Print model performance
# gbm_perf1

# Retreive test set MSE
h2o.mse(gbm_perf1) # 0.1952
h2o.mse(gbm_perf2) # 0.1954
h2o.mse(gbm_perf3) # 0.1952
h2o.mse(gbm_perf4) # 0.1948

h2o.scoreHistory(gbm_fit2) # scoring based on training set only
h2o.scoreHistory(gbm_fit3) # earlier stopping at no of trees, both training & validation perf metrics available

plot(gbm_fit3, 
     timestep = "number_of_trees", 
     metric = "Classification_error")
plot(gbm_fit3, 
     timestep = "number_of_trees", 
     metric = "logloss")

gbm_pred_p2 = h2o.predict(gbm_fit4, data_for_pred) # need to manually id class with highest prob per row

p2_output_gbm = as.data.frame(test_p2$Modelvar)
p2_output_gbm$est_var = colnames(gbm_pred_p2[, -1])[max.col(gbm_pred_p2[, -1], ties.method="first")]
colnames(p2_output_gbm) = c('act_var', 'est_var')

dim(plyr::count(p2_output_gbm$est_var)) # identified no. of variants predicted
sum(p2_output_gbm$est_var == p2_output_gbm$act_var)/nrow(p2_output_gbm) # accuracy level


# P2: h2o GBM Grid Search -------------------------------------------------

hyper_params = list(max_depth = seq(1,20,2) # usual depth = 10, max_depth = c(4,6,8,12,16,20) faster for larger datasets
                    # learn_rate = 0.05, # seq(0.05, 0.1, 0.01), # smaller learning rate is better
                    ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
                                 
                    # learn_rate_annealing = 0.99 # learning rate annealing: learning_rate shrinks by 1% after every tree 
                    ## (use 1.00 to disable, but then lower the learning_rate)
                    
                    
                    # sample_rate = seq(0.2, 1, 0.01), # sample % of rows per tree
                    # col_sample_rate = seq(0.2, 1, 0.01), # sample % of columns per split
                    # col_sample_rate_per_tree = seq(0.2, 1, 0.01)
                    # col_sample_rate_change_per_level = seq(0.9, 1.1, 0.01), # col sampling / split as a function of split depth
                    # min_rows = 2^seq(0, log2(nrow(train))-1, 1), # number of min rows in terminal node
                    # nbins = 2^seq(4, 10, 1), # no. of bins for split-finding for cont/int cols
                    # nbins_cats = 2^seq(4, 12, 1), # for cat col
                    # min_split_improvement = c(0, 1e-8, 1e-6, 1e-4) # min relative error improvement thresholds for a split to occur
                    # histogram_type = c("UniformAddptive", 'QuantilesGlobal', 'RoundRobin') # QG, RR good for num col with outliers
)

search_criteria <- list(strategy = "RandomDiscrete", # 'Cartesian'
                        max_runtime_secs = 600,
                        # max_models = 5,
                        stopping_rounds = 5,
                        stopping_metric = 'AUTO',
                        stopping_tolerance = 1e-3
)

system.time(grid <- h2o.grid(
                            hyper_params = hyper_params,
                            search_criteria = search_criteria,
                            algorithm="gbm",
                            grid_id="depth_grid", # identifier for the grid, to later retrieve it
                            
                            x = x, 
                            y = y, 
                            training_frame = train, 
                            validation_frame = valid,
                            
                            ntrees = 10000, ## more trees is better if the learning rate is small enough 
                            ## here, use "more than enough" trees - we have early stopping
                            
                            seed = 1, 
                            learn_rate = 0.01,
                            score_tree_interval = 10 # score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
                            )
)

grid@summary_table # default ordered by logloss
## sort the grid models by preferred metric
sortedGrid <- h2o.getGrid(grid@grid_id, sort_by="mse", decreasing = FALSE)
h2o.mse(h2o.performance(h2o.getModel(sortedGrid@model_ids[[1]]))) # 0.1537

## find the range of max_depth for the top 5 models - can be used to set for further tuning
topDepths = sortedGrid@summary_table$max_depth[1:5]
minDepth = min(as.numeric(topDepths))
maxDepth = max(as.numeric(topDepths))
minDepth
maxDepth

# get metric of top 5 models
for (i in 1:5) {
  gbm <- h2o.getModel(sortedGrid@model_ids[[i]])
  print(h2o.mse(h2o.performance(gbm, valid = TRUE)))
}

gbm <- h2o.getModel(sortedGrid@model_ids[[1]])
gbm@parameters # ntrees = 10000
h2o.mse(h2o.performance(gbm, newdata = test)) # check perf with test set

gbm_pred_p2 = h2o.predict(gbm, data_for_pred) # need to manually id class with highest prob per row

p2_output_gbm = as.data.frame(test_p2$Modelvar)
p2_output_gbm$est_var = colnames(gbm_pred_p2[, -1])[max.col(gbm_pred_p2[, -1], ties.method="first")]
colnames(p2_output_gbm) = c('act_var', 'est_var')

dim(plyr::count(p2_output_gbm$est_var)) # identified no. of variants predicted
sum(p2_output_gbm$est_var == p2_output_gbm$act_var)/nrow(p2_output_gbm) # accuracy level

# blending technique
prob = NULL
k=10
for (i in 1:k) { # avg of 10 models in grid search
  gbm <- h2o.getModel(sortedGrid@model_ids[[i]])
  if (is.null(prob)) prob = h2o.predict(gbm, test)
  else prob = prob + h2o.predict(gbm, test)
}
prob <- prob/k
head(prob)


# P2: h2o Deep Learning ---------------------------------------------------
# Ref: http://htmlpreview.github.io/?https://github.com/ledell/sldm4-h2o/blob/master/sldm4-deeplearning-h2o.html
# Ref: https://www.slideshare.net/0xdata/h2o-world-top-10-deep-learning-tips-tricks-arno-candel?from_action=save

library(h2o)
h2o.init(min_mem_size="4g", max_mem_size = "8g")

train = as.h2o(train_p2)
test = as.h2o(test_p2)

h2o.describe(train)
h2o.describe(test)

y <- 'Modelvar'
x <- setdiff(names(df), y)

dl_fit1 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit1",
                            # distribution = can be set
                            hidden = c(20,20), # default is c(200,200), i.e. 2 hidden layers, with 200 neurons
                            seed = 1) # not reproducibe if ran on multi core

dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit2",
                            epochs = 50, # default 10, more will increase perf, but may overfit
                            hidden = c(20,20),
                            stopping_rounds = 0,  # disable early stopping
                            seed = 1)

dl_fit3 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit3",
                            epochs = 50,
                            hidden = c(20,20),
                            nfolds = 3,                            #used for early stopping
                            score_interval = 1,                    #used for early stopping
                            stopping_rounds = 5,                   #used for early stopping
                            stopping_metric = "misclassification", #used for early stopping
                            stopping_tolerance = 1e-3,             #used for early stopping
                            # score_validation_samples=N #for sampling validation dataset 
                            # score_validation_samplings = 'Stratified' #for multi-class / imbalanced
                            # l1=1e-4, l2=1e-4, hidden_dropout_ratio = [0.2, 0.3] #use regularization
                            seed = 1)

# perform hyperparameter search
# random / grid search
# hidden (2-5 layers, 10-2000 neurons per layer)
# l1/ l2
# adaptive_rate: true (rho, epsilon), false (rate, rate_annealing, rate_decay, momentum_start, momentum_stable, momentum_ramp)

# if data is sparse / categorical predictors:
# build tiny DL model, use h2o.deepfeatures() to extract lower-dim features
# use random projection of categorical features into N-dim space, by 'max_categorical_features=N'
# use GLRM to reduce dim of dataset
# set 'sparse=True'

dl_perf1 <- h2o.performance(model = dl_fit1, newdata = test)
dl_perf2 <- h2o.performance(model = dl_fit2, newdata = test)
dl_perf3 <- h2o.performance(model = dl_fit3, newdata = test)

# Retreive test set MSE
h2o.mse(dl_perf1) # 0.2225
h2o.mse(dl_perf2) # 0.2085
h2o.mse(dl_perf3) # 0.2057

# utility functions to inspect models
h2o.scoreHistory(dl_fit3)
h2o.confusionMatrix(dl_fit3)

dl_fit3@parameters
plot(dl_fit3,
     timestep = "epochs",
     metric = "classification_error")

# Get the CV models from the `dl_fit3` object
cv_models <- sapply(dl_fit3@model$cross_validation_models, 
                    function(i) h2o.getModel(i$name))

# Plot the scoring history over time
plot(cv_models[[1]], 
     timestep = "epochs", 
     metric = "classification_error")

dl_pred_p2 = h2o.predict(dl_fit3, data_for_pred) # need to manually id class with highest prob per row

p2_output_dl = as.data.frame(test_p2$Modelvar)
p2_output_dl$est_var = p2_output_dl[, 1]
colnames(p2_output_dl) = c('act_var', 'est_var')

dim(plyr::count(p2_output_dl$est_var)) # identified no. of variants predicted
sum(p2_output_dl$est_var == p2_output_dl$act_var)/nrow(p2_output_dl) # accuracy level


# P2: DL Grid Search ----------------------------------------------------------

activation_opt <- c("Rectifier", 'RectifierWithDropuout', "Maxout", "Tanh")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)

hyper_params <- list(activation = activation_opt, l1 = l1_opt, l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 600)

dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = train,
                    validation_frame = valid,
                    seed = 1,
                    hidden = c(20,20),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria
)

dl_gridperf <- h2o.getGrid(grid_id = 'dl_grid',
                           sort_by = 'logloss'
                           # decreasing = TRUE
)

print(dl_gridperf)
dl_gridperf@summary_table

best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)
best_dl@allparameters

best_dl_perf <- h2o.performance(model = best_dl, newdata = test)
h2o.mse(best_dl_perf) # 0.2179 (logloss), 0.2154 (mse)

# P2: DL Predict --------------------------------------------------------------

dl_pred_p2 = h2o.predict(best_dl, data_for_pred) # need to manually id class with highest prob per row

p2_output_dl = as.data.frame(test_p2$Modelvar)
p2_output_dl$est_var = colnames(dl_pred_p2[, -1])[max.col(dl_pred_p2[, -1], ties.method="first")]
colnames(p2_output_dl) = c('act_var', 'est_var')

dim(plyr::count(p2_output_dl$est_var)) # identified no. of variants predicted
sum(p2_output_dl$est_var == p2_output_dl$act_var)/nrow(p2_output_dl) # accuracy level

# write.csv(p2_output_dl, file = 'p2_dl.csv')


# P2: blending (avg prob from 4 models) -----------------------------------

colnames(glm_pred_p2) == colnames(rf_pred_p2)
colnames(glm_pred_p2) == colnames(gbm_pred_p2)
colnames(glm_pred_p2) == colnames(dl_pred_p2)

data_prob_glm = as.data.frame(glm_pred_p2[ , -1])
data_prob_rf = as.data.frame(rf_pred_p2[ , -1])
data_prob_gbm = as.data.frame(gbm_pred_p2[ , -1])
data_prob_dl = as.data.frame(dl_pred_p2[ , -1])

for (i in (1:nrow(glm_pred_p2))){
  for (j in (1:ncol(glm_pred_p2))){
    avg_prob_p2[i,j] = mean(data_prob_glm[i,j], data_prob_rf[i,j], data_prob_gbm[i,j], data_prob_dl[i,j])
  }
}

for (i in (1:nrow(glm_pred_p2))){
  for (j in (1:ncol(glm_pred_p2))){
    print(c(i, j))
  }
}


ncol(glm_pred_p2)

  
mean(glm_pred_p2[1,2], rf_pred_p2[1,2], gbm_pred_p2[1,2], dl_pred_p2[1,2])
gbm_pred_p2[1,2]


# P2: h20 AutoML(Classification Issues)-----------------------------------------------------------------

str(train_p2)
df = train_p2

library(h2o)
h2o.init(min_mem_size="2g", max_mem_size = "4g")
h2o.shutdown()

df <- as.h2o(df)

h2o.describe(df)

y <- 'Modelvar'
# y <- 'Res.Price'
x <- setdiff(names(df), y)

splits <- h2o.splitFrame(df, ratios = c(0.7, .15) , seed = 1)
train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

aml_p2 <- h2o.automl(x = x,
                     y = y,
                     training_frame = train,
                     nfolds = 5,
                     keep_cross_validation_predictions = TRUE,
                     validation_frame = valid,
                     leaderboard_frame = test,
                     # exclude_algos = "GBM", # exclude_algos = c("GLM", "DeepLearning", "GBM", DRF", "StackedEnsemble"),
                     max_runtime_secs = 600, # max_models
                     seed = 1
                     # project_name = "p2_final_price"
)

print(aml_p2@leaderboard)

# Predictions

df_test <- as.h2o(test_p2)s

pred <- h2o.predict(aml_p2, df_test) # Issue: multi level classification return more data than total obs

p2_h2o_est = as.vector(pred)

# tabled predictions
p2_h2o_tbl = cbind('Actual Var' = as.character(test_p2$Modelvar), 
                   'Est Var' = p2_h2o_est
)


write.csv(p2_h2o_tbl, file = 'p2_h2o.csv')

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
train_tyt = dplyr::filter(training_data, Brand=="TOYOTA")
test_tyt = dplyr::filter(testing_data, Brand=="TOYOTA")

train_tyt$CC_scl = as.numeric(scale(train_tyt$CC_adj))
test_tyt$CC_scl = as.numeric(scale(test_tyt$CC_adj))

# filter training set only for variants with size >20
train_tyt = train_tyt %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>20) %>%
  as.data.frame()

train_tyt = droplevels(train_tyt)
test_tyt = droplevels(test_tyt)

# standardize Variant levels
tyt_var = unique(c(levels(train_tyt$Modelvar),levels(test_tyt$Modelvar)))
train_tyt$Modelvar = factor(train_tyt$Modelvar, levels = tyt_var)
test_tyt$Modelvar = factor(test_tyt$Modelvar, levels = tyt_var)

summary(train_tyt$Modelvar)
summary(test_tyt$Modelvar)

# standardize Model levels
tyt_mdl = unique(c(levels(train_tyt$Model),levels(test_tyt$Model)))
train_tyt$Model = factor(train_tyt$Model, levels = tyt_mdl)
test_tyt$Model = factor(test_tyt$Model, levels = tyt_mdl)

summary(train_tyt$Model)
summary(test_tyt$Model)

# standardize MfgYr levels
tyt_yr = unique(c(levels(train_tyt$MfgYr), levels(test_tyt$MfgYr)))
train_tyt$MfgYr = factor(train_tyt$MfgYr, levels = tyt_yr)
test_tyt$MfgYr = factor(test_tyt$MfgYr, levels = tyt_yr)

summary(train_tyt$MfgYr)
summary(test_tyt$MfgYr)



# Toyota: identify predictors ---------------------------------------------
to_drop = as.character()
for (i in 9:(length(colnames(train_tyt))-2)){
  test = plyr::count(train_tyt, colnames(train_tyt)[i])
  if (length(test$freq) == 1){ 
    # alternative we can add: if length(test$freq) > 1, then test$freq[1]/test$freq[2] < 5% then remove
    to_drop = c(paste(colnames(train_tyt)[i]), to_drop)
  }
}

drops = c(to_drop, "ID", "Brand", "Features", "Price", "CC", "CC_adj")

# drop all non-relevant predictors
train_tyt = train_tyt[, !names(train_tyt) %in% drops]
test_tyt = test_tyt[, !names(test_tyt) %in% drops]

colnames(train_tyt) == colnames(test_tyt)
colnames(train_tyt)

# SVM: Toyota -------------------------------------------------------------
system.time(tune_radial_tyt <- tune(svm, Modelvar ~ ., data = train_tyt,
                                    kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)

dim(train_tyt)

svm_tyt = svm(Modelvar ~., data = train_tyt,
              kernel="radial", cost=tune_radial_tyt$best.parameters$cost,
              gamma=tune_radial_tyt$best.parameters$gamma, trace=F
)

svm_est_tyt = predict(svm_tyt, newdata=test_tyt)

tyt_est_lvl = unique(c(levels(svm_est_tyt), levels(test_tyt$Modelvar)))
test_tyt$Modelvar = factor(test_tyt$Modelvar, levels = tyt_est_lvl)
svm_est_tyt = factor(svm_est_tyt, levels = tyt_est_lvl)

1-sum(test_tyt$Modelvar==svm_est_tyt)/nrow(test_tyt)

tyt_tbl = table(test_tyt$Modelvar, svm_est_tyt)
write.csv(tyt_tbl,"tyt.csv")

c(colnames(train_tyt), dim(train_tyt))

# Error info:
# 17.4% (incl MfgYr, CC)
# 30.4% (excl MfgYr, CC)

# 23k obs, 11 var
# user   system  elapsed 
# 19091.54    58.72 19366.04 ~322 min ~5 hrs




# Proton (P1): All -------------------------------------------------------------

train_p1 = dplyr::filter(training_data, Brand=="PROTON")
test_p1 = dplyr::filter(testing_data, Brand=="PROTON")

train_p1$CC_scl = as.numeric(scale(train_p1$CC_adj))
test_p1$CC_scl = as.numeric(scale(test_p1$CC_adj))

# filter training set only for variants with size >20
train_p1 = train_p1 %>% 
  group_by(Modelvar) %>%
  dplyr::filter(n()>20) %>%
  as.data.frame()

train_p1 = droplevels(train_p1)
test_p1 = droplevels(test_p1)

# standardize Variant levels
p1_var = unique(c(levels(train_p1$Modelvar),levels(test_p1$Modelvar)))
train_p1$Modelvar = factor(train_p1$Modelvar, levels = p1_var)
test_p1$Modelvar = factor(test_p1$Modelvar, levels = p1_var)

summary(train_p1$Modelvar)
summary(test_p1$Modelvar)

# standardize Model levels
p1_mdl = unique(c(levels(train_p1$Model),levels(test_p1$Model)))
train_p1$Model = factor(train_p1$Model, levels = p1_mdl)
test_p1$Model = factor(test_p1$Model, levels = p1_mdl)

summary(train_p1$Model)
summary(test_p1$Model)

# standardize MfgYr levels
p1_yr = unique(c(levels(train_p1$MfgYr), levels(test_p1$MfgYr)))
train_p1$MfgYr = factor(train_p1$MfgYr, levels = p1_yr)
test_p1$MfgYr = factor(test_p1$MfgYr, levels = p1_yr)

summary(train_p1$MfgYr)
summary(test_p1$MfgYr)



# P1: Identify predictors-----------------------------------------------------
to_drop = as.character()
for (i in 9:(length(colnames(train_p1))-2)){
  test = plyr::count(train_p1, colnames(train_p1)[i])
  if (length(test$freq) == 1){ 
    # alternative we can add: if length(test$freq) > 1, then test$freq[1]/test$freq[2] < 5% then remove
    to_drop = c(paste(colnames(train_p1)[i]), to_drop)
  }
}

drops = c(to_drop, "ID", "Brand", "Features", "Price", "CC", "CC_adj")

# drop all non-relevant predictors
train_p1 = train_p1[, !names(train_p1) %in% drops]
test_p1 = test_p1[, !names(test_p1) %in% drops]

colnames(train_p1) == colnames(test_p1)
colnames(train_p1)




# P1: SVM -----------------------------------------------------------------
dim(train_p1)

system.time(tune_radial_p1 <- tune(svm, Modelvar ~ ., data = train_p1,
                                   kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(0.1:5)))
)



svm_p1 = svm(Modelvar ~., data = train_p1,
             kernel="radial", cost=tune_radial_p1$best.parameters$cost,
             gamma=tune_radial_p1$best.parameters$gamma, trace=F
)

svm_est_p1 = predict(svm_p1, newdata=test_p1)

p1_est_lvl = unique(c(levels(svm_est_p1), levels(test_p1$Modelvar)))
test_p1$Modelvar = factor(test_p1$Modelvar, levels = p1_est_lvl)
svm_est_p1 = factor(svm_est_p1, levels = p1_est_lvl)

1-sum(test_p1$Modelvar==svm_est_p1)/nrow(test_p1)

p1_tbl = table(test_p1$Modelvar, svm_est_p1)
write.csv(p1_tbl,"p1.csv")

c(colnames(train_p1), dim(train_p1))

# 20k obs, 11 var
# user   system  elapsed 
# 16577.06    76.80 16814.12 

# 10,000 obs, (19-5) var
# user  system elapsed 
# 8450.45   21.45 8853.12 ~147 mins

# 10,000 obs, (17-5) var
# user  system elapsed 
# 5458.87   17.33 6132.82 ~102 mins

# Error info:
# 16% (incl MfgYr, CC)
# 32% (took out NAs in Model, excl MfgYr, CC)






# REF ---------------------------------------------------------------------

# SVM Example: https://afit-r.github.io/svm

# manual edit factor levels
levels(df$Modelvar)[levels(df$Modelvar)%in%c("1.3 (A) EZi", "1.3 (A) EZI")] = "1.3 (A) EZi"

# manual n-fold CV method:
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

CV_values_svm

# Consolidate all outputs into one file (Example) -------------------------------------------------

pred_output = cbind.data.frame(test_set$ID, pred_var)
names(pred_output) = c("ID", "EstModelVar")

# align col names
names(myvi_test_output) = c("ID", "EstModelVar")
names(city_out) = c("ID", "EstModelVar")
names(vios_test_out) = c("ID", "EstModelVar")

pred_all = rbind.data.frame(city_out, myvi_test_output, vios_test_out)
write.csv(pred_all, "pred_all.csv")






# glmnet multinomial ------------------------------------------------------

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
