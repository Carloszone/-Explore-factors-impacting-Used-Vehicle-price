library(readr)
library(dplyr)
library(tidyr)
library(scales)
library(xgboost)
library(broom)

set.seed(9527)
data <- read_csv("https://media.githubusercontent.com/media/Carloszone/ALY-6040/master/ALY6040%20dataset%20(from%20Kaggle).csv")

# Week 1
# data cleansing
## Descriptive analysis
da.res <- data %>% psych::describe() 
write.csv(da.res, file = "week 1 descriptive analysis table.csv")

### replace missing value with the mode of corresponding variable 
data$fuel[which(is.na(data$fuel))] = names(sort(-table(data$fuel)))[1]
data$title_status[which(is.na(data$title_status))] = names(sort(-table(data$title_status)))[1]

## outliers of variables
out.res <- data %>% select(price, odometer) %>% psych::describe() 
write.csv(out.res, file = "week 1 out.result table.csv")

### removing outliers which are greater than mean +2*sd
while(sum(data$price > mean(data$price) + 2 * sd(data$price)) != 0){
  index <- which(data$price > mean(data$price) + 2 * sd(data$price))
  data <- data[-index,]
}

while(sum(data$odometer > mean(data$odometer) + 2 * sd(data$odometer)) != 0){
  index <- which(data$odometer > mean(data$odometer) + 2 * sd(data$odometer))
  data <- data[-index,]
}

# Week 2
# linear model
## calculate the age of car
data <- data  %>% mutate(age = 2020- year)

## recode factor variables
US <- c("buick", "chrysler", "harley-davidson", "lincoln", "mercury", "cadillac",
        "chevrolet", "dodge", "ford", "gmc", "jeep", "pontiac", "ram", "saturn")
JP <- c("acura", "infiniti", "nissan", "toyota", "datsun", "honda", "lexus", "mazda",
        "mitsubishi", "subaru")
UK <- c("rover", "jaguar", "land rover", "mini")
DE <- c("audi", "bmw", "mercedes-benz", "volkswagen")
IT <- "fiat"
KR <- c("kia", "hyundai")
SE <- "volvo"

small <- c("convertible", "coupe", "hatchback", "mini-van", "sedan")
large <- c("offroad", "pickup", "SUV", "truck", "van", "wagon", "bus")

classic <- c("white", "black", "silver", "grey")
other <- c("blue", "brown", "green", "orange", "purple", "red", "yellow")

#new manufacturer types
data$newmanu[data$manufacturer %in% US] <- "US"
data$newmanu[data$manufacturer %in% JP] <- "JP"
data$newmanu[data$manufacturer %in% UK] <- "UK"
data$newmanu[data$manufacturer %in% DE] <- "DE"
data$newmanu[data$manufacturer %in% IT] <- "IT"
data$newmanu[data$manufacturer %in% KR] <- "KR"
data$newmanu[data$manufacturer %in% SE] <- "SE"

#new title types
data$newtitle[data$title_status != "clean"] <- "not"
data$newtitle[data$title_status == "clean"] <- "clean"

#new type types
data$newtype[data$type %in% small] <- "small"
data$newtype[data$type %in% large] <- "large"
data$newtype[data$type == "other"] <- "other"

# new color
data$newcolor[data$paint_color %in% classic] <- "classic"
data$newcolor[data$paint_color %in% other] <- "other"
data$newcolor[data$paint_color == "custom"] <- "custom"


## set dummy variable
dummy_manu <- model.matrix(~newmanu, data)
dummy_fuel <- model.matrix(~fuel, data)
dummy_title <- model.matrix(~newtitle, data)
dummy_type <- model.matrix(~newtype, data)
dummy_color <- model.matrix(~newcolor, data)


## combine the model dataset
model.data <- cbind(data[,c(4,8,15)], dummy_manu, dummy_fuel, dummy_title,
                    dummy_type, dummy_color)

x <- model.data[,-c(4,11,16,18,21)] ##delete the "(Intercept)" columns

## linear model
model.original <- lm(price~.,x)

## save regression result
res <- tidy(lm(price~.,x), conf.int = T)
write.csv(res, file = "linear model coefficients result.csv")

# Week 3
## variable derivement
data$manucountry[data$newmanu == "US"] <- "America"
data$manucountry[data$newmanu %in% c("JP", "KR")] <- "Asia"
data$manucountry[data$newmanu %in% c("UK", "DE", "IT", "SE")] <- "Europe"

data$Fossil_Fuel[data$fuel %in% c("gas", "diesel")] <- 1
data$Fossil_Fuel[data$fuel %in% c("electric", "hybrid", "other")] <- 0

## transform to dummy variable
dummy_manu <- model.matrix(~manucountry, data)

## create model data set
model.data <- cbind(data[,c(4,8,15,21)], dummy_manu, dummy_title, dummy_type, dummy_color)

x <- model.data[,-c(5,8,10,13)] ##delete the "(Intercept)" columns

index <- sample((1:nrow(x)), round(0.8*nrow(x)))
training <- x[index,]
testing <- x[-index,]

## build the saturated model
model.saturated <- lm(price~.,data = training)
model.inter <- lm(price~1, data = training)

## model optimization method: selection
model.aic.back <- MASS::stepAIC(model.saturated, direction = "backward", trace = 0)
model.aic.forw <- MASS::stepAIC(model.inter, direction = "forward",
                                scope = list(lower = model.inter, 
                                             upper = ~ + odometer +
                                               age + Fossil_Fuel + manucountryAsia +
                                               manucountryEurope + newtitlenot + 
                                               newtypeother + newtypesmall + newcolorcustom +
                                               newcolorother))

## model optimization method: gradient boosting
model.gb <- xgboost(data = as.matrix(training[,-1]), label = as.vector(training[,1]),
                    booster = "gblinear", nround = 200)
xgb.importance(model = model.gb)

## calculate the MSE for each model
MSE.saturated.train <- sum((training[,1] - predict(model.saturated, training))^2)/length(training[,1])
MSE.selection.train <- sum((training[,1] - predict(model.aic.back, training))^2)/length(training[,1])
MSE.gb.train <- sum((training[,1] - predict(model.gb, as.matrix(training[,-1])))^2)/length(training[,1])

MSE.saturated <- sum((testing[,1] - predict(model.saturated, testing))^2)/length(testing[,1])
MSE.selection <- sum((testing[,1] - predict(model.aic.back, testing))^2)/length(testing[,1])
MSE.gb <- sum((testing[,1] - predict(model.gb, as.matrix(testing[,-1])))^2)/length(testing[,1])

#Week 4
## add set type tag
data$set_type[index] <- "training set"
data$set_type[-index] <- "testing set"

## add prediction results
data$prediction[index] <- predict(model.gb, as.matrix(training[,-1]))
data$prediction[-index] <- predict(model.gb, as.matrix(testing[,-1]))



## create a new data set
col_index <- c(4,8,15,17:23)
data_for_tableau <- data[,col_index]

### set new group tag for NO.
data_for_tableau$group <- 72
group <- seq(0, nrow(data), by = 100)

for(n in 1:71){
  a = group[n]
  b = group[n]+99
  data_for_tableau$group[a:b] <- n
}

### set new bins for age and odometer
data_for_tableau <- data_for_tableau %>% 
  mutate(age_group = (age %/% 10 + 1) * 10) %>%
  mutate(odo_group = (odometer %/% 5000 + 1) * 5000)


write.csv(data_for_tableau, file = "data for tableau.csv")

# calculate R2
MSE.gb.all <- sum((x[,1] - predict(model.gb, as.matrix(x[,-1])))^2)/length(x[,1])

r2.gb.all <- 1- MSE.gb.all/var(x[,1])
r2.gb.train <- 1 - MSE.gb.train/var(training[,1])
r2.gb <- 1 - MSE.gb/var(testing[,1])
