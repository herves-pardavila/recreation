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
import geopandas as gpd
plt.close("all")

if __name__== "__main__":
   
    path=r"F:/doctorado/"
    #load the data
    #df=pd.read_csv(path+"3travel_cost_Ons.csv")
    df=pd.read_csv(path+"recreation/ZonalTravelCost/datos_provinciales/3travel_cost_Ons_ready.csv")
    df=df[df.Lugar != "Pontevedra"]
    df=df[df.Año.isin([2022,2023])]
    df=df.groupby(["Lugar","Zona"],as_index=False).mean(numeric_only=True)
    #remove nans
    

    

    variable="turistasINE"
    df.dropna(subset=[variable]+[ "Lugar", "distance (km)", "Población","median_inc"], inplace = True)
    #set data types
    df.Año=df.Año.astype("category")
    df.Lugar=df.Lugar.astype("category")
    #df["Numero"]=df.Numero.astype(int)
    df.turistasINE=df.turistasINE.astype(int)
    df.Población=df.Población.astype(int)
    df.median_inc=df.median_inc.astype(float)
    df.TC=df.TC.astype(float)
    df["median_inc2"]=df.median_inc*df.median_inc
  
    print(df)
    print(df.info())
    
    #summary statistics
    sum_statistics=df[["turistasINE","median_inc","TC"]].describe()
    print(sum_statistics)
    #correlations
    print(df[["median_inc","distance (km)","TC"]].corr("spearman",numeric_only=True))

    
    df["y"]=df[variable]
    df["logy"]=np.log(df.y)
    df["Vrate"]=1000*df.y/df.Población
    df["lnVrate"]=np.log(df.Vrate)
    df["pop"]=df.Población
    df["lnpop"]=np.log(df.Población)
    df["lnTC"]=np.log(df.TC)
    df["lnI"]=np.log(df.median_inc)
    df["Y"]=df.y
    
    # #homocedasticity test
    # sigmas=[]
    # for i in range(100):
    #     sigmas+=[np.var(np.random.permutation(df.lnVrate)[0:10])]
    # fig1=plt.figure()
    # ax1=fig1.add_subplot(121)
    # ax1.set_ylabel("Varianza")
    # ax2=fig1.add_subplot(122)
    # ax2.set_ylabel("Probabilidad")
    # ax1.plot(sigmas)
    # ax1.hlines(y=np.var(df.lnVrate), xmin=0, xmax=100, linewidth=2, color='r')
    
    # #normality test
    # mu=np.mean(df.lnVrate)
    # sigma=np.std(df.lnVrate)
    # xt=np.linspace(-10,5,1000)
    # yt=np.exp(-1*(xt-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    # ax2.hist(df.lnVrate,density=True,bins=7)
    # ax2.plot(xt,yt)
  
    
    
    
    df.loc[df.Zona=="Europa","Zona"]="Mundo"
    df.Zona=df.Zona.astype("category")
    #np.random.seed(seed=1)
    #mask=np.random.rand(len(df))<0.999
    #df_train=df[mask]
    df_train=df
    #df_test=df[~mask]


    #overdispersion
    print("La sobredispersion es del",df.Y.mean()/df.Y.std())
    
    #poisson model
    model="log-log"
    expr="""y~lnTC + lnI"""
    null_expr="Y~1"
  
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    #y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson(),exposure=df["pop"]).fit()
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
    nb1=nb1.fit(method="nm",maxiter=50000,maxfun=50000)
    print(nb1.summary())
    #print("AIC=",nb2_training_results.aic)
    
    if model == "log-lin":
        CS=-1/(nb1.params[1]) #modelo log-lig
        sCS=((1/nb1.params[1])**2)*nb1.bse[1] #+ 2* ((1/nb1.params[1])**6)*nb1.bse[1]**4
        
    elif model == "log-log":
        CS=-1*df.TC.mean(skipna=True)/(nb1.params[1]+1) #modelo log-log
        sCS=((df.TC.mean()/((nb1.params[1]+1)**2)))*nb1.bse[1]
        
    
    print("Consumer Surplus= %f (%f)" %(CS,sCS))
   
    
   #merge new variables with the geodatabase
   
   
    gdf=gpd.read_file(path +"recreation/ZonalTravelCost/datos_provinciales/3travel_cost_Ons.gpkg")
    newdf=pd.merge(gdf,df[["Lugar","CT_(€)", "OC_(€)","TC","Vrate"]],how="left",on="Lugar")
    
    new_gdf=gpd.GeoDataFrame(data=newdf,crs=gdf.crs,geometry=newdf.geometry)
    new_gdf.to_file(path +"recreation/ZonalTravelCost/datos_provinciales/3travel_cost_Ons_ready.gpkg")
    

        
    
    
    
    
    




   
