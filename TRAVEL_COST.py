import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
import statsmodels as st
from sklearn.metrics import r2_score
from scipy import stats
from datetime import datetime

if __name__== "__main__":
    
    #load the data
    df=pd.read_csv("3travel_cost_ready.csv")
    #remove nans
    #df.dropna(subset=["Lugar","Numero","distance (km)","Población","Income"],inplace=True)

    


    #set data types
    
    df.Numero=df.Numero.astype(int)
    df.Población=df.Población.astype(int)
    df.median_inc=df.median_inc.astype(float)
    df["median_inc2"]=df.median_inc*df.median_inc
    print(df)
    print(df.info())
    
    

    #summary statistics
    sum_statistics=df.describe()
    print(sum_statistics)
    #correlations
    print(df.corr("spearman"))

    df["y"]=df.Numero
    #divide between training set and test set
    df["Vrate"]=1*df.y/df.Población
    df["logVrate"]=np.log(df.Vrate)
    df["logy"]=np.log(df.y)
    df.loc[df.Zona=="Europa","Zona"]="Mundo"
    df.Zona=df.Zona.astype("category")
    #np.random.seed(seed=1)
    #mask=np.random.rand(len(df))<0.999
    #df_train=df[mask]
    df_train=df
    #df_test=df[~mask]

    #poisson model
    
    print(df)
    
    
    expr="""y~TC +median_inc"""
    null_expr="y~1"
    
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    #y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_results.summary())
    print("AIC=",poisson_training_results.aic)
    #print("Mean mu=",poisson_training_results.mu)
    print("Consumer Surplus=",-1/poisson_training_results.params[1])

    #auxiliary regression model
    df_train['BB_LAMBDA'] = poisson_training_results.mu
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['y'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA-1"""
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    print(aux_olsr_results.summary())
    print("Value of alpha=",aux_olsr_results.params[0])

    #NB1 regression
    print("=========================Negative Binomial Regression===================== ")
    #exog=sm.add_constant(X_train)

    y_train=y_train.iloc[:,0]
    nb1=sm.GeneralizedPoisson(y_train,X_train.iloc[:,:]
                              )
    nb1=nb1.fit()
    print(nb1.summary())
    #print("AIC=",nb2_training_results.aic)

    # #intercept only poisson regression to compute pseudoR2
    # y_train, X_train = dmatrices(null_expr, df_train, return_type='dataframe')
    # y_test, X_test = dmatrices(null_expr, df_test, return_type='dataframe')
    # poisson_null_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    # print("Deviance of intercept only model=",poisson_null_results.deviance)
    # print("Deviance of full model=",poisson_training_results.deviance)
    # print("Pseudo R2 deviance",1-(poisson_training_results.deviance/poisson_null_results.deviance))



    
    
    expr="""Vrate~ TC + median_inc2"""
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    print(X_train)
   # y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    ols_training_results = sm.OLS(y_train, X_train, family=sm.families.Gaussian()).fit()
    print(ols_training_results.summary())
    print("Consumer Surplus=",-1/ols_training_results.params[1])

