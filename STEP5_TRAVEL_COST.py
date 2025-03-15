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
from matplotlib import cm
plt.close("all")

if __name__== "__main__":
    
    #load the data
    df=pd.read_csv("3travel_cost_Ons.csv")
    df=df[df.Año.isin([2022,2023])]
    #remove nans
    df.dropna(subset=[ "Lugar", "Numero", "distance (km)", "Población"], inplace = True)

    


    #set data types
    df.Año=df.Año.astype("category")
    df.Lugar=df.Lugar.astype("category")
    df["turistasParque"]=df.Numero.astype(int)
    df.turistasINE=df.turistasINE.astype(int)
    df.Población=df.Población.astype(int)
    df.median_inc=df.median_inc.astype(float)
    df["median_inc2"]=df.median_inc*df.median_inc
  
    print(df)
    print(df.info())
    
    #summary statistics
    sum_statistics=df[["Numero","turistasINE","median_inc","TC"]].describe()
    print(sum_statistics)
    #correlations
    print(df[["median_inc","distance (km)","TC"]].corr("spearman",numeric_only=True))

    variable="turistasINE"
    df["y"]=df[variable]
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
    ax1.set_ylabel("Varianza")
    ax2=fig1.add_subplot(122)
    ax2.set_ylabel("Probabilidad")
    ax1.plot(sigmas)
    ax1.hlines(y=np.var(df.lnVrate), xmin=0, xmax=100, linewidth=2, color='r')
    
    #normality test
    mu=np.mean(df.lnVrate)
    sigma=np.std(df.lnVrate)
    xt=np.linspace(-10,5,1000)
    yt=np.exp(-1*(xt-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    ax2.hist(df.lnVrate,density=True,bins=7)
    ax2.plot(xt,yt)
  
    
    
    
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
    model="log-lin"
    expr="""y~TC + median_inc"""
    null_expr="Y~1"
  
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    #y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_results.summary())
    print("AIC=",poisson_training_results.aic)
    #print("Mean mu=",poisson_training_results.mu)
    

    #auxiliary regression model
    df_train['BB_LAMBDA'] = poisson_training_results.mu
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Y'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) /1, axis=1)
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
    nb1=sm.NegativeBinomialP(y_train,X_train.iloc[:,:],p=1,exposure=np.array(df["pop"]))
    #nb1=sm.NegativeBinomial(y_train,X_train.iloc[:,:],loglike_method="nb1",exposure=np.array(df["pop"]))
    nb1=nb1.fit(method="nm",maxiter=50000,maxfun=50000)
    print(nb1.summary())
    #print("AIC=",nb2_training_results.aic)
    
    if model == "log-lin":
        CS=-1/(nb1.params[1]) #modelo log-lin
    elif model == "log-log":
        CS=-1*df.TC.mean()/(nb1.params[1]+1) #modelo log-log
    print("Consumer Surplus=",CS)
    
    predictions=nb1.predict(X_train)
    
    
    # #compute psuedo-R2
    # #negative binomial regression with intercept only to compute pseudo R2 using deviance
    # y_train_null, X_train_null = dmatrices(null_expr, df_train, return_type='dataframe')
    # nb1_intercept_only= sm.NegativeBinomialP(y_train_null, X_train_null,p=1,exposure=np.array(df["pop"]))
    # nb1_intercept_only=nb1_intercept_only.fit(method="nm",maxiter=50000,maxfun=50000)
    # pseudoR2=1-(nb1.deviance/nb1_intercept_only.deviance)
   
    

   
    fig2=plt.figure()
    fig2.suptitle("Demand Curve for %s model with %s \n CS=%f €" %(model,
                                                                    variable,CS),fontsize=15)
    ax=fig2.add_subplot(111)
    fig2.subplots_adjust(left=0.15,bottom=0.15,top=0.8)
    ax.set_ylabel("Travel Cost (€)",fontsize=15)
    ax.set_xlabel("Visitation Rate per 1000 habitants",fontsize=15)
    #df.sort_values(by=["Vrate","TC"],inplace=True)
    ax.plot(df.Vrate,df.TC,"o",label="observed")
    ax.plot(1000*predictions/df.Población,df.TC,"o",label="predicted")
    plt.tick_params(axis='both', which='both', labelsize=15)
    
    path_for_figures="/media/david/EXTERNAL_USB/doctorado/recreation/ZonalTravelCost/"
    fig2.savefig(path_for_figures+model+variable+"Carnota.png")
    fig2.savefig(path_for_figures+model+variable+"Carnota.pdf")
    fig2.legend(loc="center right",fontsize=20)
    
    #calcular la superficie teórica Q=Q(TC,I)
    vector_TC=np.linspace(df.TC.min(),df.TC.max(), 100)
    vector_I=np.linspace(df.median_inc.min(),df.median_inc.max(),100)
    X,Y=np.meshgrid(vector_TC,vector_I)
    
    Z=np.exp(nb1.params[0]+nb1.params[1]*X+nb1.params[2]*Y)
    
    fig3=plt.figure()
    ax3=fig3.add_subplot(projection="3d")
    
    surf = ax3.plot_surface(X, Y, 1000*Z,color="red",alpha=0.6)
    ax3.set_xlabel("Travel Cost")
    ax3.set_ylabel("Income")
    ax3.set_zlabel("VR (per 1000 habitants)")
    
    #calcular la superficie Q=Q(TC,I) experimental 
    
    VR=1000*predictions/df.Población
    surf = ax3.plot_trisurf(X_train.TC, X_train.median_inc, VR, color="blue", alpha=0.6)
   
    

        
    
    
    
    
    




   
