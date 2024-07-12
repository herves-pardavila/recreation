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
    
    #load the data
    dfINE=pd.read_csv("tourist_origins_distances.csv")
    print(dfINE)
    #remove rows with no tourist data
  

    # #set data types
    # df.Date=df.Date.astype("category")
    # df.Month=df.Month.astype("category")
    # df.Year=df.Year.astype("category")
    # df.Season=df.Season.astype("category")
    # df.covid=df.covid.astype("category")
    # df.turistas=df.turistas.astype(int)
    # #df.Poblacion=df.Poblacion.astype(int)
    # df["distance"]=df["distance (km)"].astype(float)
    # print(df)
    # print(df.info())
    # #remove nans
    # df=df[["Date","origen","turistas","distance","Poblacion","Income","Season","covid"]]
    # df.dropna(subset=["origen","turistas","distance","Poblacion","Income"],inplace=True)
    
    # #summary statistics
    # sum_statistics=df[["turistas","distance","Poblacion"]].describe()
    # print(sum_statistics)
    # #correlations
    # print(df[["turistas","distance","Poblacion"]].corr("spearman"))

    # #divide between training set and test set
    # df.turistas=df.turistas.astype(int)
    # df["Vrate"]=100*df.turistas/df.Poblacion
    # np.random.seed(seed=1)
    # mask=np.random.rand(len(df))<0.999
    # df_train=df[mask]
    # df_train=df
    # df_test=df[~mask]

    # #poisson model
    
    # print(df.Vrate.unique())
    
    
    # expr="""turistas~ + distance + Season + covid + Income"""
    # # # #null_expr="""Visitantes ~ IdOAPN + Season + covid """
    
    # y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    # y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    # poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    # print(poisson_training_results.summary())
    # print("AIC=",poisson_training_results.aic)
    # print("Mean mu=",poisson_training_results.mu)

    # #auxiliary regression model
    # df_train['BB_LAMBDA'] = poisson_training_results.mu
    # df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['turistas'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
    # ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA -1"""
    # aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    # print(aux_olsr_results.summary())
    # print("Value of alpha=",aux_olsr_results.params[0])

    # #negative_binomial regression
    # print("=========================Negative Binomial Regression===================== ")
    # nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= aux_olsr_results.params[0] )).fit()
    # summary=nb2_training_results.summary()
    # print(summary)
    # print("AIC=",nb2_training_results.aic)
   
    # print(df.Poblacion.unique())
