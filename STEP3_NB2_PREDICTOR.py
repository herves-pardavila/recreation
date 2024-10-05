import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
from sklearn.metrics import r2_score
from scipy import stats
from datetime import datetime

if __name__== "__main__":
    
    #load the INE  data
    dfINE=pd.read_csv("INE_data.csv")
    print(dfINE)
    #set data types
    dfINE["turistasINE"]=dfINE.Numero.astype(int)
    dfINE.Zona=dfINE.Zona.astype("category")
    #load the real data
    df=pd.read_csv("data_original.csv")
    df.Numero=df.Numero.astype(int)
    df.Zona=df.Zona.astype("category")
    print(df)

    #merge both
    df=pd.merge(df,dfINE[["Lugar","turistasINE","Zona","Año"]],on=["Lugar","Zona","Año"],how="inner")
    print("================================================================================================")
    #df=df[df.Año == 2019]
    print(df)

    #poisson model
    
    expr="""Numero~ turistasINE """
    null_expr="""Numero ~ 1 """
    
    y_train, X_train = dmatrices(expr, df, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df, return_type='dataframe')
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_results.summary())
    # print("AIC=",poisson_training_results.aic)
    # print("Mean mu=",poisson_training_results.mu)

    #auxiliary regression model
    df['BB_LAMBDA'] = poisson_training_results.mu
    df['AUX_OLS_DEP'] = df.apply(lambda x: ((x['Numero'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / 1, axis=1)
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA -1"""
    aux_olsr_results = smf.ols(ols_expr, df).fit()
    print(aux_olsr_results.summary())
    print("Value of alpha=",aux_olsr_results.params[0])

    #negative_binomial regression
    print("=========================Negative Binomial (p=1) Regression===================== ")
    nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= aux_olsr_results.params[0] )).fit()
    summary=nb2_training_results.summary()
    print(summary)
    print("AIC=",nb2_training_results.aic)

    nb1=sm.NegativeBinomialP(y_train,X_train.iloc[:,:],p=1)
    nb1=nb1.fit(method="nm",maxiter=5000,maxfun=5000,start_params=[1,1,aux_olsr_results.params[0]])
    print(nb1.summary())
   
    # # print(df.Poblacion.unique())

    #predctions
    #nb2_predictions=nb2_training_results.get_prediction(X_train)
    #prediction_summary_frame=nb2_predictions.summary_frame()
    nb1_predictions=nb1.predict(X_train)
    df["yhat_full"]=nb1_predictions
    #df["yhat_full"]=prediction_summary_frame["mean"]

    # #regression with intercept only to compute pseudo R2 using deviance
    # y_train, X_train = dmatrices(null_expr, df, return_type='dataframe')
    # intercept_only= sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
    # df["R2dev"]=1-(nb2_training_results.deviance/intercept_only.deviance)
    # print(df)
    print(df[["Año","Zona","Lugar","Numero","turistasINE","yhat_full"]])

    # print(df[df.Lugar=="Galicia"].yhat_full.sum())
    # print(df.yhat_full.sum())
    df[["Año","Lugar","Zona","Numero","turistasINE","yhat_full","median_inc","Población","distance (km)"]].to_csv("3travel_cost.csv",index=False)
    
