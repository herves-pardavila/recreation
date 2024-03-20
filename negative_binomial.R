library(foreign)
library(ggplot2)
library(MASS)
df <- read.csv("/home/usuario/Documentos/recreation/recreation.csv")
df <- within(df,{
    Date <- factor(Date)
    Month <- factor(Month)
    SITE_NAME <- factor(SITE_NAME)
    Season <- factor(Season)
    logPUD <- log(PUD+1)
    log_turistas_corregido <- log(turistas_corregido+1)
    log_turistas_total <- log(turistas_total+1)
})
print(summary(df$Month))
print(mean(df$Visitantes)/sqrt(var(df$Visitantes)))

#model <- glm.nb(Visitantes  ~ turistas_total + Season + SITE_NAME,  data=df)

#print(summary(model))