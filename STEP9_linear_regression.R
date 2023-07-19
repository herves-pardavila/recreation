#library(rgdal)

df <- read.csv("PUD.csv")
df <- df[,names(df)!="geometry"]
#df$protected_<- as.factor(df$protected)
df$logPUD <- log(df$PUD+1)
print(summary(df))


formula <- "logPUD ~ dist.train+trial_len + hotels+ Com.nodes..m.+protected+dem+HSOL_SUM_1+PRED_AVG_1+TA_AVG_1.5+TA_MAX_1.5+TA_MIN_1.5+PP_SUM_1.5+VV_AVG_2m+viewpoint+beach+restaurant+bike_len+nautic+lighthouse..m.+town..m.+city..m.+arable+built+crops+forest+grassland+intertidal+marshes+rocks+scrub"
formula2 <- "logPUD ~trial_len + hotels+ Com.nodes..m.+protected+dem+PRED_AVG_1+viewpoint+beach+restaurant+nautic+lighthouse..m.+built+crops+forest+grassland+rocks+scrub"
model<-lm(formula,df)
print(summary(model))
