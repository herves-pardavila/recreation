library(xtable)

df <- read.csv("PUD.csv")
df <- na.omit(df)
df[df$PUD %in% 2:5,"PUD"]<-2
df[df$PUD>5,"PUD"]<-3
#df$PUD <- df$PUD+1
df$Month <- as.factor(df$Month)
#df$PUD <- as.factor(df$PUD)
df$FID <- as.factor(df$FID)
print(summary(df))

explanatory_variables <- c("TA_MAX_1.5m","TA_MIN_1.5m","PP_SUM_1.5m","VV_AVG_2m","HSOL_SUM_1.5m","PR_AVG_1.5m","trial_len","bike_len","Com.nodes..m.","dist.train",
"lighthouse..m.","city..m.","restaurant","viewpoint","hotel","Protected.area..m2.","Beach.area..m2.","Port.area..m2.","Area..km2.", "Month")
formula <- paste(explanatory_variables,collapse="+")
formula <- paste("PUD ~",formula, collpase=" ")

# formula2 <- "logPUD ~trial_len + hotels+ Com.nodes..m.+protected+dem+PRED_AVG_1+viewpoint+beach+restaurant+nautic+lighthouse..m.+built+crops+forest+grassland+rocks+scrub"


set.seed(19)
sample <- sample.int(n=nrow(df),size=floor(0.7*nrow(df)),replace=F)
train <- df[sample, ]
test <- df[-sample,c("PUD",explanatory_variables)]

model<-glm(formula,train,family="poisson")
print(summary(model))
# table <- summary(model)
# print(xtable(table,type="latex"),file="results.tex")

# probabilities <-predict(model,test[,explanatory_variables],type="response")
# contrasts(test$PUD)
# predicted.classes <- ifelse(probabilities > 0.5, 1, 0)




