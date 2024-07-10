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
    
    #========================================== DATA WITH INFORMATION ABOUT ORIGINS =====================================

    #load the data
    newdf=pd.read_csv("travel_cost.csv")
    #remove rows with no tourist data
    bad_index=newdf[(newdf.turistas.isin([" ","."]) )].index
    newdf=newdf[~newdf.index.isin(bad_index)]

    #set data types
    newdf.Date=newdf.Date.astype("category")
    newdf.Month=newdf.Month.astype("category")
    newdf.Year=newdf.Year.astype("category")
    newdf.Season=newdf.Season.astype("category")
    newdf.covid=newdf.covid.astype("category")
    newdf.turistas=newdf.turistas.astype(int)
    #newdf.Poblacion=newdf.Poblacion.astype(int)
    newdf["distance"]=newdf["distance (km)"].astype(float)
    print(newdf)
    print(newdf.info())
    #remove nans
    newdf=newdf[["Date","origen","turistas","distance","Poblacion","Income","Season","covid"]]
    newdf.dropna(subset=["origen","turistas","distance","Poblacion","Income"],inplace=True)
    
    #summary statistics
    sum_statistics=newdf[["turistas","distance","Poblacion","Income"]].describe()
    print(sum_statistics)
    #correlations
    print(newdf[["turistas","distance","Poblacion","Income"]].corr("spearman"))

    newdf.rename(columns={"turistas":"turistas_total"},inplace=True)
    newdf["Visitantes"]=1
    newdf["IdOAPN"]="Islas Atlánticas de Galicia"


    # =================================================== DATA FOR TRAINING THE NEGATIVE BINOMILA FUNCTION ================================================
    #prepare the data
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation_ready.csv")
    df=df[df.IdOAPN=="Islas Atlánticas de Galicia"]
    df.Date=df.Date.astype("category")
    df.Month=df.Month.astype("category")
    df.Year=df.Year.astype("category")
    df.IdOAPN=df.IdOAPN.astype("category")
    df.Season=df.Season.astype("category")
    df.covid=df.covid.astype("category")

    #divide between training set and test set
    df.dropna(subset=["Visitantes","turistas_total"],inplace=True)
    df.Visitantes=df.Visitantes.astype(int)
    df.turistas_total=df.turistas_total.astype(int)
    df_train=df


    #poisson model
  
    expr2="""Visitantes ~ turistas_total +IdOAPN+ Season + covid"""

    expr=expr2
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_results.summary())
    print("AIC=",poisson_training_results.aic)
    print("Mean mu=",poisson_training_results.mu)


    #auxiliary regression model
    df_train['BB_LAMBDA'] = poisson_training_results.mu
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Visitantes'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA -1"""
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    print(aux_olsr_results.summary())
    print("Value of alpha=",aux_olsr_results.params[0])


    #negative_binomial regression
    print("=========================Negative Binomial Regression===================== ")
    nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= aux_olsr_results.params[0] )).fit()
    summary=nb2_training_results.summary()
    print(summary)
    print("AIC=",nb2_training_results.aic)


    #=============================== USE THE DATASET WITH ORIGIN INFORMATION TO MAKE PREDICTIONS ===========================

    y_test, X_test = dmatrices(expr, newdf, return_type='dataframe')
    nb2_predictions = nb2_training_results.get_prediction(X_test)
    predictions_summary_frame = nb2_predictions.summary_frame()
    newdf["yhat"]=predictions_summary_frame['mean']

    print(newdf.describe())
    print(newdf)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    
    # expr="""Vrate ~ + distance + Season + covid + Income"""
    # # # #null_expr="""Visitantes ~ IdOAPN + Season + covid """
    
    # y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    # y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    # ols_training_results = sm.OLS(y_train, X_train, family=sm.families.Gaussian()).fit()
    # print(ols_training_results.summary())
    # print("AIC=",ols_training_results.aic)

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
