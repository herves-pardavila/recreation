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
plt.close("all")
if __name__== "__main__":
    #load the data
    df=pd.read_csv("3travel_cost_aiguestortes.csv")
    #remove nans
    #df.dropna(subset=["Lugar","Numero","distance (km)","Población","Income"],inplace=True)

    


    #set data types
    df.Año=df.Año.astype("category")
    df.Lugar=df.Lugar.astype("category")
    df.Numero=df.Numero.astype(int)
    df.turistasINE=df.turistasINE.astype(int)
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
    print(df[["median_inc","distance (km)","TC"]].corr("spearman",numeric_only=True))

    
    df["y"]=df.Numero
    df["logy"]=np.log(df.y)
    df["Vrate"]=1000*df.y/df.Población
    df["lnVrate"]=np.log(df.Vrate)
    df["pop"]=df.Población
    df["lnpop"]=np.log(df.Población)
    df["lnTC"]=np.log(df.TC)
    df["lnI"]=np.log(df.median_inc)
    df["Y"]=df.y
    
    #homocedasticity test
    sigmas=[]
    for i in range(100):
        sigmas+=[np.var(np.random.permutation(df.lnVrate)[0:10])]
    fig1=plt.figure()
    ax1=fig1.add_subplot(121)
    ax2=fig1.add_subplot(122)
    ax1.plot(sigmas)
    ax1.hlines(y=np.var(df.lnVrate), xmin=0, xmax=100, linewidth=2, color='r')
    
    #normality test
    mu=np.mean(df.lnVrate)
    sigma=np.std(df.lnVrate)
    xt=np.linspace(-5,5,1000)
    yt=np.exp(-1*(xt-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    ax2.plot(xt,yt)
    ax2.hist(df.lnVrate,density=True,histtype="step",bins=5)
    
  
    
    
    
    df.loc[df.Zona=="Europa","Zona"]="Mundo"
    df.Zona=df.Zona.astype("category")
    #np.random.seed(seed=1)
    #mask=np.random.rand(len(df))<0.999
    #df_train=df[mask]
    df_train=df
    #df_test=df[~mask]


    #overdispersion
    print("La sobredispersion es del",df.Y.mean()/df.Y.var())
    #poisson model
    expr="""y~TC +median_inc"""
    null_expr="Y~1"
  
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    #y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_results.summary())
    print("AIC=",poisson_training_results.aic)
    #print("Mean mu=",poisson_training_results.mu)
    

    # #auxiliary regression model
    # df_train['BB_LAMBDA'] = poisson_training_results.mu
    # df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Y'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) /1, axis=1)
    # ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA -1"""
    # aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    # print(aux_olsr_results.summary())
    # print("Value of alpha=",aux_olsr_results.params[0])

    #NB1 regression
    print("========================= Negative Binomial 1 Regression ===================== ")
    #exog=sm.add_constant(X_train)

    y_train=y_train.iloc[:,0]
    #print(y_train)
    #print(X_train)
    nb1=sm.NegativeBinomialP(y_train,X_train.iloc[:,:],p=1,exposure=np.array(df["pop"]))
    nb1=nb1.fit(method="nm",maxiter=50000,maxfun=50000)
    print(nb1.summary())
    CS=-1/(nb1.params[1])
    print("Consumer Surplus=",CS)
   #  print("Consumer Surplus Lineal =",-1/nb1.params[1])
   #  #print("AIC=",nb2_training_results.aic)

   #  nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
   #  print(nb2_training_results.summary())
   #  #intercept only poisson regression to compute pseudoR2
   #  y_train, X_train = dmatrices(null_expr, df_train, return_type='dataframe')
   #  #y_test, X_test = dmatrices(null_expr, df_test, return_type='dataframe')
   #  null_results = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
   #  print("Pseudo R2 deviance",1-(nb2_training_results.deviance/null_results.deviance))



    
    
   #  expr="""Vrate~ TC + median_inc """
   #  y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
   #  #print(X_train)
   #  #print(y_train)
   # # y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
   #  beta_model = BetaModel(y_train, X_train).fit(method="nm",maxiter=50000,maxfun=50000)
   #  print(beta_model.summary())
   #  print("Consumer Surplus=",((df.Vrate.median()**(-2))/(df.Población.median()*beta_model.params[1]*(-1+1/(df.Vrate.median())))))
   #  print("Consumer Surplus Lineal =",-1/beta_model.params[1])


    # ols_expr="""lnVrate~ lnTC + lnI"""
    # ols_model = smf.ols(ols_expr, df_train ).fit()
    # print(ols_model.summary())
    # CS=float(-1*df.TC.mean()/(ols_model.params[1]+1))
    # #CS=(-1/(ols_model.params[1]))
    # print("Consumer Surplus por persona=",CS)
   
