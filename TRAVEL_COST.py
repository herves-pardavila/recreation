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
    df=pd.read_csv("tourist_origins_distances.csv")
    #remove nans
    df.dropna(subset=["Lugar","Numero","distance (km)","Población","Income"],inplace=True)

    


    #set data types
    df.Año=df.Año.astype("category")
    df.Numero=df.Numero.astype(int)
    df.Población=df.Población.astype(int)
    df.Income=df.Income.astype(float)
    df.Income=df.Income/1000
    df["distance"]=df["distance (km)"].astype(float)
    print(df)
    print(df.info())
    
    

    #summary statistics
    sum_statistics=df.describe()
    print(sum_statistics)
    #correlations
    print(df.corr("spearman"))

    #divide between training set and test set
    df["Vrate"]=df.Numero/df.Población
    df["logVrate"]=np.log(df.Vrate)
    df["logY"]=np.log(df.Numero)
    np.random.seed(seed=1)
    mask=np.random.rand(len(df))<0.999
    df_train=df[mask]
    df_train=df
    df_test=df[~mask]

    #poisson model
    
    print(df.Vrate.unique())
    
    
    expr="""Numero~ + distance + Income"""
    
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_results.summary())
    print("AIC=",poisson_training_results.aic)
    print("Mean mu=",poisson_training_results.mu)

    # #auxiliary regression model
    # df_train['BB_LAMBDA'] = poisson_training_results.mu
    # df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Numero'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
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

    
    
    expr="""logVrate ~ + distance + Income"""
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    ols_training_results = sm.OLS(y_train, X_train, family=sm.families.Gaussian()).fit()
    print(ols_training_results.summary())
