import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
from sklearn.metrics import r2_score
from scipy import stats
if __name__== "__main__":

    #prepare the data
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation_ready.csv")
    #df=df[df.SITE_NAME=="Teide National Park"]
    df.Date=df.Date.astype("category")
    df.Month=df.Month.astype("category")
    df.Year=df.Year.astype("category")
    df.SITE_NAME=df.SITE_NAME.astype("category")
    df.Season=df.Season.astype("category")
    #df.Visitantes=df.Visitantes-1
    df["logVisitantes"]=np.log(df.Visitantes+1)
    print(df)

    #divide between training set and test set
    df.dropna(subset=["Visitantes","PUD"],inplace=True)
    df.Visitantes=df.Visitantes.astype(int)
    #df.PUD=df.PUD.astype(int)
    #df.turistas_total=df.turistas_total.astype(int)
    df["log_Summer_turistas_corregido"]=np.log(df.Summer_turistas_corregido+1)
    #df["log_Summer_turistas"]=np.log(df.Summer_turistas+1)
    print(df.info())
    np.random.seed(seed=1)
    mask=np.random.rand(len(df))<0.8
    df_train=df[mask]
    #df_train=df
    df_test=df[~mask]

    #poisson model
    expr1="""logVisitantes ~ logPUD + SITE_NAME + Season """
    expr2="""logVisitantes ~ log_turistas+ SITE_NAME + Season + log_Summer_turistas_corregido"""
    expr=expr1
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    training_results = sm.GLM(y_train, X_train, family=sm.families.Gaussian()).fit()
    print(training_results.summary())
    predictions=training_results.predict(X_test)
    df_test["yhat"]=predictions
    res=stats.cramervonmises_2samp(df_test.logVisitantes,df_test.yhat)
    print(res)
    print(res.pvalue > 0.05)



    #chi2=np.sum(((df_test.Visitantes-df_test.yhat)**2)/df_test.yhat)
    #print(chi2)
    #print(len(df_test))
    #print("Mean mu=",poisson_training_results.mu)
