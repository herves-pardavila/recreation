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
from statsmodels.othermod.betareg import BetaModel

if __name__== "__main__":
    
    #load the data
    df=pd.read_csv("3travel_cost_ons.csv")
    #remove nans
    #df.dropna(subset=["Lugar","Numero","distance (km)","Población","Income"],inplace=True)

    


    #set data types
    df.Año=df.Año.astype("category")
    df.Numero=df.Numero.astype(int)
    df.Población=df.Población.astype(int)
    df.median_inc=df.median_inc.astype(float)
    df["median_inc2"]=df.median_inc*df.median_inc
    #df=df[df.Año.isin([2019])]
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
    df["logI"]=np.log(df["median_inc"])
    df["logI2"]=np.log(df["median_inc"]**2)
    df.loc[df.Zona=="Europa","Zona"]="Mundo"
    df.Zona=df.Zona.astype("category")
    #np.random.seed(seed=1)
    #mask=np.random.rand(len(df))<0.999
    #df_train=df[mask]
    df_train=df
    #df_test=df[~mask]


    #overdispersion
    print("La sobredispersion es del",df.y.mean()/df.y.var())
    #poisson model
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
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['y'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) /1, axis=1)
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA -1"""
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    print(aux_olsr_results.summary())
    print("Value of alpha=",aux_olsr_results.params[0])

    #NB1 regression
    print("========================= Negative Binomial 1 Regression ===================== ")
    #exog=sm.add_constant(X_train)

    y_train=y_train.iloc[:,0]
    #print(y_train)
    #print(X_train)
    nb1=sm.NegativeBinomialP(y_train,X_train.iloc[:,:],p=1)
    nb1=nb1.fit(method="nm",maxiter=50000,maxfun=50000)
    print(nb1.summary())
    print("Consumer Surplus=",-1/(nb1.params[1]*y_train.mean()))
    print("Consumer Surplus Lineal =",-1/nb1.params[1])
    #print("AIC=",nb2_training_results.aic)

    nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
    print(nb2_training_results.summary())
    #intercept only poisson regression to compute pseudoR2
    y_train, X_train = dmatrices(null_expr, df_train, return_type='dataframe')
    #y_test, X_test = dmatrices(null_expr, df_test, return_type='dataframe')
    null_results = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
    print("Pseudo R2 deviance",1-(nb2_training_results.deviance/null_results.deviance))



    
    
    expr="""Vrate~ TC + median_inc """
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    #print(X_train)
    #print(y_train)
   # y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    beta_model = BetaModel(y_train, X_train).fit(method="nm",maxiter=50000,maxfun=50000)
    print(beta_model.summary())
    print("Consumer Surplus=",((df.Vrate.median()**(-2))/(df.Población.median()*beta_model.params[1]*(-1+1/(df.Vrate.median())))))
    print("Consumer Surplus Lineal =",-1/beta_model.params[1])


    ols_expr="""logy~ TC + median_inc"""
    ols_model = smf.ols(ols_expr, df_train ).fit()
    print(ols_model.summary())
    print("Consumer Surplus=",-1/(ols_model.params[1]*df.y.mean()))
    print("Consumer Surplus Lineal =",-1/ols_model.params[1])

