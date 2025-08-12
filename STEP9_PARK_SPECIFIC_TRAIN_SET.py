import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime
from matplotlib.ticker import MaxNLocator, FuncFormatter
plt.close("all")

def scientific_notation(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.log10(abs(x)))
    coefficient = x / 10**exponent
    return f"{coefficient:.0f}·10$^{exponent}$"

def mape(df,model):
     df=df[["Visitantes",model]]
     print(df)
     df.dropna(inplace=True)
     print(df)
     n=len(df)
     return (100/n)*np.sum(np.abs((df.Visitantes-df[model])/df.Visitantes))
def mae(df,model):
     df=df[["Visitantes",model]]
     print(df)
     df.dropna(inplace=True)
     print(df)
     n=len(df)
     return (1/n)*np.sum(np.abs((df.Visitantes-df[model])))   
     
def ajuste(df,expr,name):
    log_likelihood=[]
    newdf=pd.DataFrame()
    null_expr="""Visitantes ~ 1 """
    for park in df.IdOAPN.unique():
        #print("===========================================")
        #print("Park=",park)
        #print("===========================================")
        subdf=df[df.IdOAPN==park]

        #divide between training set and test set
        # np.random.seed(seed=1)
        # mask=np.random.rand(len(subdf))<0.7
        df_train=subdf[subdf.Year.isin([2021,2022])]
        df_test=subdf[subdf.Year==2023]

        y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
        y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
        try:
            poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

            #auxiliary regression model
            df_train['BB_LAMBDA'] = poisson_training_results.mu
            df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Visitantes'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
            ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""
            aux_olsr_results = smf.ols(ols_expr, df_train).fit()

            #negative_binomial regression
            #print("=========================Negative Binomial Regression===================== ")
            nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
            #predicitons
            nb2_predictions = nb2_training_results.get_prediction(X_test)
            predictions_summary_frame = nb2_predictions.summary_frame()
            df_test["yhat"+name]=predictions_summary_frame['mean']

            #negative binomial regression with intercept only to compute pseudo R2 using deviance
            y_train, X_train = dmatrices(null_expr, df_train, return_type='dataframe')
            nb2_intercept_only= sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= aux_olsr_results.params[0] )).fit()
            print(df_test)
            df_test["R2dev"+name]=mape(df_test,"yhat"+name)
            #df_test["R2dev"+name]=1-(nb2_training_results.deviance/nb2_intercept_only.deviance)
            
            newdf=pd.concat([newdf,df_test])

        except ZeroDivisionError:
                continue
    log_likelihood=np.array(log_likelihood)
    print("Suma de los log-likelihood=",np.sum(log_likelihood))
    newdf=newdf[["Date","IdOAPN","Visitantes"]+["yhat"+name]+["R2dev"+name]]
    newdf=newdf.round({"R2dev"+name:2})
    return newdf
if __name__== "__main__":
    
    main_path="/home/david/Documents/"
    #prepare the data
    dataframe=pd.read_csv(main_path+"recreation/recreation_ready.csv")
    dataframe.Date=dataframe.Date.astype("category")
    dataframe.Month=dataframe.Month.astype("category")
    dataframe.Year=dataframe.Year.astype("category")
    dataframe.IdOAPN=dataframe.IdOAPN.astype("category")
    dataframe.Season=dataframe.Season.astype("category")
    dataframe.covid=dataframe.covid.astype("category")
    dataframe.Visitantes=dataframe.Visitantes.astype(int)
    print(dataframe.info()) 
    
    #model 1
    expr1="""Visitantes ~ logPUD + Season + covid - 1"""
    df1=dataframe.dropna(subset=["Visitantes","PUD"])
    newdf1=ajuste(df1,expr1,"model1")
    print(newdf1)
    
    #model2
    expr2="""Visitantes ~ turistas_total+ Season + covid -1 """
    df2=dataframe.dropna(subset=["Visitantes","turistas_total"])
    df2.turistas_total=df2.turistas_total.astype(int)
    newdf2=ajuste(df2,expr2,"model2")
    
    # #model4
    # expr4="""Visitantes ~ logPUD + logIUD + Season + covid -1 """
    # df4=dataframe.dropna(subset=["Visitantes","IUD","PUD"])
    # df4["logIUD"]=np.log(df4.IUD+1)
    # newdf4=ajuste(df4,expr4,"model4")

    #model 5
    expr5="""Visitantes ~ logPUD + turistas_total+ Season + covid -1 """
    df5=dataframe.dropna(subset=["Visitantes","turistas_total","PUD"])
    df5.turistas_total=df5.turistas_total.astype(int)
    newdf5=ajuste(df5,expr5,"model5")
   

   
    
    newdf=pd.merge(newdf1,newdf2,on=["Date","IdOAPN","Visitantes"],how="outer")
    #newdf=pd.merge(newdf,newdf4,on=["Date","IdOAPN","Visitantes"],how="outer")
    newdf=pd.merge(newdf,newdf5,on=["Date","IdOAPN","Visitantes"],how="outer")
    print(newdf)

    newdfA=newdf[newdf.IdOAPN=="Aigüestortes i Estany de Sant Maurici"]
    newdfB=newdf[newdf.IdOAPN=="Archipiélago de Cabrera"]
    newdfC=newdf[newdf.IdOAPN=="Cabañeros"]
    newdfD=newdf[newdf.IdOAPN=="Caldera de Taburiente"]
    newdfE=newdf[newdf.IdOAPN=="Doñana"]
    newdfF=newdf[newdf.IdOAPN=="Garajonay"]
    newdfG=newdf[newdf.IdOAPN=="Islas Atlánticas de Galicia"]
    newdfH=newdf[newdf.IdOAPN=="Monfragüe"]
    newdfI=newdf[newdf.IdOAPN=="Ordesa y Monte Perdido"]
    newdfJ=newdf[newdf.IdOAPN=="Picos de Europa"]
    newdfK=newdf[newdf.IdOAPN=="Sierra Nevada"]
    newdfL=newdf[newdf.IdOAPN=="Sierra de Guadarrama"]
    newdfM=newdf[newdf.IdOAPN=="Sierra de las Nieves"]
    newdfN=newdf[newdf.IdOAPN=="Tablas de Daimiel"]
    newdfO=newdf[newdf.IdOAPN=="Teide National Park"]
    newdfP=newdf[newdf.IdOAPN=="Timanfaya"]

   

    fig2=plt.figure(figsize=(20,30))
    ax2=fig2.subplot_mosaic("""ABCD
              EFGH
              IJKL
              MNOP
              """)
      
    fig2.subplots_adjust(hspace=0.2,wspace=0.175,left=0.09,right=0.95, top=0.92, bottom=0.085)
    fig2.text(x=0.45,y=0.025,s="Observed Visitors",fontsize=35)
    fig2.text(x=0.025,y=0.4,s="On-site Visitors",rotation="vertical",fontsize=35)
   
    ax2["A"].plot(newdfA.Visitantes/newdfA.Visitantes.max(),newdfA.yhatmodel1/newdfA.Visitantes.max(),"o",color="red",label="Flickr",markersize=10)
    ax2["A"].plot(newdfA.Visitantes/newdfA.Visitantes.max(),newdfA.yhatmodel2/newdfA.Visitantes.max(),"o",color="green",label="Phones",markersize=10)
    #ax2["A"].plot(newdfA.Visitantes/newdfA.Visitantes.max(),newdfA.yhatmodel4/newdfA.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["A"].plot(newdfA.Visitantes/newdfA.Visitantes.max(),newdfA.yhatmodel5/newdfA.Visitantes.max(),"o",color="black",label="Flickr+Phones",markersize=10)
    ax2["A"].plot(np.linspace(min(newdfA.Visitantes)/newdfA.Visitantes.max(),max(newdfA.Visitantes)/newdfA.Visitantes.max(),100),
      np.linspace(min(newdfA.Visitantes)/newdfA.Visitantes.max(),max(newdfA.Visitantes)/newdfA.Visitantes.max(),100),label="1-1 line",lw=5)
    ax2["A"].set_title(newdfA.IdOAPN.unique()[0],fontsize=20)
    ax2["A"].text(0.4,0.04,str(newdfA.R2devmodel1.min()),transform=ax2["A"].transAxes,color="red",fontsize=23)
    ax2["A"].text(0.6,0.04,str(newdfA.R2devmodel2.min()),transform=ax2["A"].transAxes,color="green",fontsize=23)
    #ax2["A"].text(0.8,0.05,str(newdfA.R2devmodel4.min()),transform=ax2["A"].transAxes,color="blue",fontsize=23)
    ax2["A"].text(0.8,0.04,str(newdfA.R2devmodel5.min()),transform=ax2["A"].transAxes,color="black",fontsize=23)
    #ax2["A"].tick_params(axis='x',labelsize=0)
    #ax2["A"].tick_params(axis='y',labelsize=20)
   
    ax2["B"].plot(newdfB.Visitantes/newdfB.Visitantes.max(),newdfB.yhatmodel1/newdfB.Visitantes.max(),"o",color="red",markersize=10)
    ax2["B"].plot(newdfB.Visitantes/newdfB.Visitantes.max(),newdfB.yhatmodel2/newdfB.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["B"].plot(newdfB.Visitantes/newdfB.Visitantes.max(),newdfB.yhatmodel4/newdfB.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["B"].plot(newdfB.Visitantes/newdfB.Visitantes.max(),newdfB.yhatmodel5/newdfB.Visitantes.max(),"o",color="black",markersize=10)
    ax2["B"].plot(np.linspace(min(newdfB.Visitantes)/newdfB.Visitantes.max(),max(newdfB.Visitantes)/newdfB.Visitantes.max(),100),
      np.linspace(min(newdfB.Visitantes)/newdfB.Visitantes.max(),max(newdfB.Visitantes)/newdfB.Visitantes.max(),100),lw=5)
    ax2["B"].set_title(newdfB.IdOAPN.unique()[0],fontsize=20)
    ax2["B"].text(0.4,0.04,str(newdfB.R2devmodel1.min()),transform=ax2["B"].transAxes,color="red",fontsize=23)
    ax2["B"].text(0.6,0.04,str(newdfB.R2devmodel2.min()),transform=ax2["B"].transAxes,color="green",fontsize=23)
    #ax2["B"].text(0.8,0.05,str(newdfB.R2devmodel4.min()),transform=ax2["B"].transAxes,color="blue",fontsize=23)
    ax2["B"].text(0.8,0.04,str(newdfB.R2devmodel5.min()),transform=ax2["B"].transAxes,color="black",fontsize=23)
    #ax2["B"].tick_params(axis='both',labelsize=0)
    ax2["B"].set_ylim(-0.1,2.5)
    
    ax2["C"].plot(newdfC.Visitantes/newdfC.Visitantes.max(),newdfC.yhatmodel1/newdfC.Visitantes.max(),"o",color="red",markersize=10)
    ax2["C"].plot(newdfC.Visitantes/newdfC.Visitantes.max(),newdfC.yhatmodel2/newdfC.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["C"].plot(newdfC.Visitantes/newdfC.Visitantes.max(),newdfC.yhatmodel4/newdfC.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["C"].plot(newdfC.Visitantes/newdfC.Visitantes.max(),newdfC.yhatmodel5/newdfC.Visitantes.max(),"o",color="black",markersize=10)
    ax2["C"].plot(np.linspace(min(newdfC.Visitantes)/newdfC.Visitantes.max(),max(newdfC.Visitantes)/newdfC.Visitantes.max(),100),
      np.linspace(min(newdfC.Visitantes)/newdfC.Visitantes.max(),max(newdfC.Visitantes)/newdfC.Visitantes.max(),100),lw=5)
    ax2["C"].set_title(newdfC.IdOAPN.unique()[0],fontsize=20)
    ax2["C"].text(0.4,0.04,str(newdfC.R2devmodel1.min()),transform=ax2["C"].transAxes,color="red",fontsize=23)
    ax2["C"].text(0.6,0.04,str(newdfC.R2devmodel2.min()),transform=ax2["C"].transAxes,color="green",fontsize=23)
    #ax2["C"].text(0.8,0.05,str(newdfC.R2devmodel4.min()),transform=ax2["C"].transAxes,color="blue",fontsize=23)
    ax2["C"].text(0.8,0.04,str(newdfC.R2devmodel5.min()),transform=ax2["C"].transAxes,color="black",fontsize=23)
    #ax2["C"].tick_params(axis='both',labelsize=0)

    
    ax2["D"].plot(newdfD.Visitantes/newdfD.Visitantes.max(),newdfD.yhatmodel1/newdfD.Visitantes.max(),"o",color="red",markersize=10)
    ax2["D"].plot(newdfD.Visitantes/newdfD.Visitantes.max(),newdfD.yhatmodel2/newdfD.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["D"].plot(newdfD.Visitantes/newdfD.Visitantes.max(),newdfD.yhatmodel4/newdfD.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["D"].plot(newdfD.Visitantes/newdfD.Visitantes.max(),newdfD.yhatmodel5/newdfD.Visitantes.max(),"o",color="black",markersize=10)
    ax2["D"].plot(np.linspace(min(newdfD.Visitantes)/newdfD.Visitantes.max(),max(newdfD.Visitantes)/newdfD.Visitantes.max(),100),
      np.linspace(min(newdfD.Visitantes)/newdfD.Visitantes.max(),max(newdfD.Visitantes)/newdfD.Visitantes.max(),100),lw=5)
    ax2["D"].set_title(newdfD.IdOAPN.unique()[0],fontsize=20)
    ax2["D"].text(0.4,0.04,str(newdfD.R2devmodel1.min()),transform=ax2["D"].transAxes,color="red",fontsize=23)
    ax2["D"].text(0.6,0.04,str(newdfD.R2devmodel2.min()),transform=ax2["D"].transAxes,color="green",fontsize=23)
    #ax2["D"].text(0.8,0.05,str(newdfD.R2devmodel4.min()),transform=ax2["D"].transAxes,color="blue",fontsize=23)
    ax2["D"].text(0.8,0.04,str(newdfD.R2devmodel5.min()),transform=ax2["D"].transAxes,color="black",fontsize=23)
    #ax2["D"].tick_params(axis='y',labelsize=20,right=True)
    #ax2["D"].yaxis.tick_right()
    #ax2["C"].tick_params(axis='x',labelsize=0)
    ax2["D"].set_ylim(0.3,1.8)
    
    ax2["E"].plot(newdfE.Visitantes/newdfE.Visitantes.max(),newdfE.yhatmodel1/newdfE.Visitantes.max(),"o",color="red",markersize=10)
    ax2["E"].plot(newdfE.Visitantes/newdfE.Visitantes.max(),newdfE.yhatmodel2/newdfE.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["E"].plot(newdfE.Visitantes/newdfE.Visitantes.max(),newdfE.yhatmodel4/newdfE.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["E"].plot(newdfE.Visitantes/newdfE.Visitantes.max(),newdfE.yhatmodel5/newdfE.Visitantes.max(),"o",color="black",markersize=10)
    ax2["E"].plot(np.linspace(min(newdfE.Visitantes)/newdfE.Visitantes.max(),max(newdfE.Visitantes)/newdfE.Visitantes.max(),100),
      np.linspace(min(newdfE.Visitantes)/newdfE.Visitantes.max(),max(newdfE.Visitantes)/newdfE.Visitantes.max(),100),lw=5)
    ax2["E"].set_title(newdfE.IdOAPN.unique()[0],fontsize=20)
    ax2["E"].text(0.4,0.04,str(newdfE.R2devmodel1.min()),transform=ax2["E"].transAxes,color="red",fontsize=23)
    ax2["E"].text(0.6,0.04,str(newdfE.R2devmodel2.min()),transform=ax2["E"].transAxes,color="green",fontsize=23)
    #ax2["E"].text(0.8,0.05,str(newdfE.R2devmodel4.min()),transform=ax2["E"].transAxes,color="blue",fontsize=23)
    ax2["E"].text(0.8,0.04,str(newdfE.R2devmodel5.min()),transform=ax2["E"].transAxes,color="black",fontsize=23)
    #ax2["E"].tick_params(axis='y',labelsize=20)
    #ax2["E"].tick_params(axis='x',labelsize=0)
    
    ax2["F"].plot(newdfF.Visitantes/newdfF.Visitantes.max(),newdfF.yhatmodel1/newdfF.Visitantes.max(),"o",color="red",markersize=10)
    ax2["F"].plot(newdfF.Visitantes/newdfF.Visitantes.max(),newdfF.yhatmodel2/newdfF.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["F"].plot(newdfF.Visitantes/newdfF.Visitantes.max(),newdfF.yhatmodel4/newdfF.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["F"].plot(newdfF.Visitantes/newdfF.Visitantes.max(),newdfF.yhatmodel5/newdfF.Visitantes.max(),"o",color="black",markersize=10)
    ax2["F"].plot(np.linspace(min(newdfF.Visitantes)/newdfF.Visitantes.max(),max(newdfF.Visitantes)/newdfF.Visitantes.max(),100),
      np.linspace(min(newdfF.Visitantes)/newdfF.Visitantes.max(),max(newdfF.Visitantes)/newdfF.Visitantes.max(),100),lw=5)
    ax2["F"].set_title(newdfF.IdOAPN.unique()[0],fontsize=20)
    ax2["F"].text(0.4,0.04,str(newdfF.R2devmodel1.min()),transform=ax2["F"].transAxes,color="red",fontsize=23)
    ax2["F"].text(0.6,0.04,str(newdfF.R2devmodel2.min()),transform=ax2["F"].transAxes,color="green",fontsize=23)
    #ax2["F"].text(0.8,0.05,str(newdfF.R2devmodel4.min()),transform=ax2["F"].transAxes,color="blue",fontsize=23)
    ax2["F"].text(0.8,0.04,str(newdfF.R2devmodel5.min()),transform=ax2["F"].transAxes,color="black",fontsize=23)
    #ax2["F"].tick_params(axis='both',labelsize=0)
    ax2["F"].set_ylim(0.2,1)

    ax2["G"].plot(newdfG.Visitantes/newdfG.Visitantes.max(),newdfG.yhatmodel1/newdfG.Visitantes.max(),"o",color="red",markersize=10)
    ax2["G"].plot(newdfG.Visitantes/newdfG.Visitantes.max(),newdfG.yhatmodel2/newdfG.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["G"].plot(newdfG.Visitantes/newdfG.Visitantes.max(),newdfG.yhatmodel4/newdfG.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["G"].plot(newdfG.Visitantes/newdfG.Visitantes.max(),newdfG.yhatmodel5/newdfG.Visitantes.max(),"o",color="black",markersize=10)
    ax2["G"].plot(np.linspace(min(newdfG.Visitantes)/newdfG.Visitantes.max(),max(newdfG.Visitantes)/newdfG.Visitantes.max(),100),
      np.linspace(min(newdfG.Visitantes)/newdfG.Visitantes.max(),max(newdfG.Visitantes)/newdfG.Visitantes.max(),100),lw=5)
    ax2["G"].set_title(newdfG.IdOAPN.unique()[0],fontsize=20)
    ax2["G"].text(0.4,0.04,str(newdfG.R2devmodel1.min()),transform=ax2["G"].transAxes,color="red",fontsize=23)
    ax2["G"].text(0.6,0.04,str(newdfG.R2devmodel2.min()),transform=ax2["G"].transAxes,color="green",fontsize=23)
    #ax2["G"].text(0.8,0.05,str(newdfG.R2devmodel4.min()),transform=ax2["G"].transAxes,color="blue",fontsize=23)
    ax2["G"].text(0.8,0.04,str(newdfG.R2devmodel5.min()),transform=ax2["G"].transAxes,color="black",fontsize=23)
    #ax2["G"].tick_params(axis='both',labelsize=0)

    ax2["H"].plot(newdfH.Visitantes/newdfH.Visitantes.max(),newdfH.yhatmodel1/newdfH.Visitantes.max(),"o",color="red",markersize=10)
    ax2["H"].plot(newdfH.Visitantes/newdfH.Visitantes.max(),newdfH.yhatmodel2/newdfH.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["H"].plot(newdfH.Visitantes/newdfH.Visitantes.max(),newdfH.yhatmodel4/newdfH.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["H"].plot(newdfH.Visitantes/newdfH.Visitantes.max(),newdfH.yhatmodel5/newdfH.Visitantes.max(),"o",color="black",markersize=10)
    ax2["H"].plot(np.linspace(min(newdfH.Visitantes)/newdfH.Visitantes.max(),max(newdfH.Visitantes)/newdfH.Visitantes.max(),100),
      np.linspace(min(newdfH.Visitantes)/newdfH.Visitantes.max(),max(newdfH.Visitantes)/newdfH.Visitantes.max(),100),lw=5)
    ax2["H"].set_title(newdfH.IdOAPN.unique()[0],fontsize=20)
    ax2["H"].text(0.4,0.04,str(newdfH.R2devmodel1.min()),transform=ax2["H"].transAxes,color="red",fontsize=23)
    ax2["H"].text(0.6,0.04,str(newdfH.R2devmodel2.min()),transform=ax2["H"].transAxes,color="green",fontsize=23)
    #ax2["H"].text(0.8,0.05,str(newdfH.R2devmodel4.min()),transform=ax2["H"].transAxes,color="blue",fontsize=23)
    ax2["H"].text(0.8,0.04,str(newdfH.R2devmodel5.min()),transform=ax2["H"].transAxes,color="black",fontsize=23)
    #ax2["H"].tick_params(axis='y',labelsize=20,right=True)
    #ax2["H"].yaxis.tick_right()
    #ax2["H"].tick_params(axis='x',labelsize=0)

    ax2["I"].plot(newdfI.Visitantes/newdfI.Visitantes.max(),newdfI.yhatmodel1/newdfI.Visitantes.max(),"o",color="red",markersize=10)
    ax2["I"].plot(newdfI.Visitantes/newdfI.Visitantes.max(),newdfI.yhatmodel2/newdfI.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["I"].plot(newdfI.Visitantes/newdfI.Visitantes.max(),newdfI.yhatmodel4/newdfI.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["I"].plot(newdfI.Visitantes/newdfI.Visitantes.max(),newdfI.yhatmodel5/newdfI.Visitantes.max(),"o",color="black",markersize=10)
    ax2["I"].plot(np.linspace(min(newdfI.Visitantes)/newdfI.Visitantes.max(),max(newdfI.Visitantes)/newdfI.Visitantes.max(),100),
      np.linspace(min(newdfI.Visitantes)/newdfI.Visitantes.max(),max(newdfI.Visitantes)/newdfI.Visitantes.max(),100),lw=5)
    ax2["I"].set_title(newdfI.IdOAPN.unique()[0],fontsize=20)
    ax2["I"].text(0.4,0.04,str(newdfI.R2devmodel1.min()),transform=ax2["I"].transAxes,color="red",fontsize=23)
    ax2["I"].text(0.6,0.04,str(newdfI.R2devmodel2.min()),transform=ax2["I"].transAxes,color="green",fontsize=23)
    #ax2["I"].text(0.8,0.05,str(newdfI.R2devmodel4.min()),transform=ax2["I"].transAxes,color="blue",fontsize=23)
    ax2["I"].text(0.8,0.04,str(newdfI.R2devmodel5.min()),transform=ax2["I"].transAxes,color="black",fontsize=23)
    #ax2["I"].tick_params(axis='y',labelsize=20)
    #ax2["I"].tick_params(axis='x',labelsize=0)
    
    ax2["J"].plot(newdfJ.Visitantes/newdfJ.Visitantes.max(),newdfJ.yhatmodel1/newdfJ.Visitantes.max(),"o",color="red",markersize=10)
    ax2["J"].plot(newdfJ.Visitantes/newdfJ.Visitantes.max(),newdfJ.yhatmodel2/newdfJ.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["J"].plot(newdfJ.Visitantes/newdfJ.Visitantes.max(),newdfJ.yhatmodel4/newdfJ.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["J"].plot(newdfJ.Visitantes/newdfJ.Visitantes.max(),newdfJ.yhatmodel5/newdfJ.Visitantes.max(),"o",color="black",markersize=10)
    ax2["J"].plot(np.linspace(min(newdfJ.Visitantes)/newdfJ.Visitantes.max(),max(newdfJ.Visitantes)/newdfJ.Visitantes.max(),100),
      np.linspace(min(newdfJ.Visitantes)/newdfJ.Visitantes.max(),max(newdfJ.Visitantes)/newdfJ.Visitantes.max(),100),lw=5)
    ax2["J"].set_title(newdfJ.IdOAPN.unique()[0],fontsize=20)
    ax2["J"].text(0.4,0.04,str(newdfJ.R2devmodel1.min()),transform=ax2["J"].transAxes,color="red",fontsize=23)
    ax2["J"].text(0.6,0.04,str(newdfJ.R2devmodel2.min()),transform=ax2["J"].transAxes,color="green",fontsize=23)
    #ax2["J"].text(0.8,0.05,str(newdfJ.R2devmodel4.min()),transform=ax2["J"].transAxes,color="blue",fontsize=23)
    ax2["J"].text(0.8,0.04,str(newdfJ.R2devmodel5.min()),transform=ax2["J"].transAxes,color="black",fontsize=23)
    #ax2["J"].tick_params(axis='both',labelsize=0)
    ax2["J"].set_ylim(-0.1,1.75)

    ax2["K"].plot(newdfK.Visitantes/newdfK.Visitantes.max(),newdfK.yhatmodel1/newdfK.Visitantes.max(),"o",color="red",markersize=10)
    ax2["K"].plot(newdfK.Visitantes/newdfK.Visitantes.max(),newdfK.yhatmodel2/newdfK.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["K"].plot(newdfK.Visitantes/newdfK.Visitantes.max(),newdfK.yhatmodel4/newdfK.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["K"].plot(newdfK.Visitantes/newdfK.Visitantes.max(),newdfK.yhatmodel5/newdfK.Visitantes.max(),"o",color="black",markersize=10)
    ax2["K"].plot(np.linspace(min(newdfK.Visitantes)/newdfK.Visitantes.max(),max(newdfK.Visitantes)/newdfK.Visitantes.max(),100),
      np.linspace(min(newdfK.Visitantes)/newdfK.Visitantes.max(),max(newdfK.Visitantes)/newdfK.Visitantes.max(),100),lw=5)
    ax2["K"].set_title(newdfK.IdOAPN.unique()[0],fontsize=20)
    ax2["K"].text(0.4,0.04,str(newdfK.R2devmodel1.min()),transform=ax2["K"].transAxes,color="red",fontsize=23)
    ax2["K"].text(0.6,0.04,str(newdfK.R2devmodel2.min()),transform=ax2["K"].transAxes,color="green",fontsize=23)
    #ax2["K"].text(0.8,0.05,str(newdfK.R2devmodel4.min()),transform=ax2["K"].transAxes,color="blue",fontsize=23)
    ax2["K"].text(0.8,0.04,str(newdfK.R2devmodel5.min()),transform=ax2["K"].transAxes,color="black",fontsize=23)
    #ax2["K"].tick_params(axis='both',labelsize=0)
    ax2["K"].set_ylim(0.2,1)

    ax2["L"].plot(newdfL.Visitantes/newdfL.Visitantes.max(),newdfL.yhatmodel1/newdfL.Visitantes.max(),"o",color="red",markersize=10)
    ax2["L"].plot(newdfL.Visitantes/newdfL.Visitantes.max(),newdfL.yhatmodel2/newdfL.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["L"].plot(newdfL.Visitantes/newdfL.Visitantes.max(),newdfL.yhatmodel4/newdfL.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["L"].plot(newdfL.Visitantes/newdfL.Visitantes.max(),newdfL.yhatmodel5/newdfL.Visitantes.max(),"o",color="black",markersize=10)
    ax2["L"].plot(np.linspace(min(newdfL.Visitantes)/newdfL.Visitantes.max(),max(newdfL.Visitantes)/newdfL.Visitantes.max(),100),
      np.linspace(min(newdfL.Visitantes)/newdfL.Visitantes.max(),max(newdfL.Visitantes)/newdfL.Visitantes.max(),100),lw=5)
    ax2["L"].set_title(newdfL.IdOAPN.unique()[0],fontsize=20)
    ax2["L"].text(0.4,0.04,str(newdfL.R2devmodel1.min()),transform=ax2["L"].transAxes,color="red",fontsize=23)
    ax2["L"].text(0.6,0.04,str(newdfL.R2devmodel2.min()),transform=ax2["L"].transAxes,color="green",fontsize=23)
    #ax2["L"].text(0.8,0.05,str(newdfL.R2devmodel4.min()),transform=ax2["L"].transAxes,color="blue",fontsize=23)
    ax2["L"].text(0.8,0.04,str(newdfL.R2devmodel5.min()),transform=ax2["L"].transAxes,color="black",fontsize=23)
    #ax2["L"].tick_params(axis='x',labelsize=0)
    #ax2["L"].tick_params(axis='y',labelsize=20)
    #ax2["L"].yaxis.tick_right()

    ax2["M"].plot(newdfM.Visitantes/newdfM.Visitantes.max(),newdfM.yhatmodel1/newdfM.Visitantes.max(),"o",color="red",markersize=10)
    ax2["M"].plot(newdfM.Visitantes/newdfM.Visitantes.max(),newdfM.yhatmodel2/newdfM.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["M"].plot(newdfM.Visitantes/newdfM.Visitantes.max(),newdfM.yhatmodel4/newdfM.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["M"].plot(newdfM.Visitantes/newdfM.Visitantes.max(),newdfM.yhatmodel5/newdfM.Visitantes.max(),"o",color="black",markersize=10)
    ax2["M"].plot(np.linspace(min(newdfM.Visitantes)/newdfM.Visitantes.max(),max(newdfM.Visitantes)/newdfM.Visitantes.max(),100),
      np.linspace(min(newdfM.Visitantes)/newdfM.Visitantes.max(),max(newdfM.Visitantes)/newdfM.Visitantes.max(),100),lw=5)
    ax2["M"].set_title(newdfM.IdOAPN.unique()[0],fontsize=20)
    ax2["M"].text(0.4,0.04,str(newdfM.R2devmodel1.min()),transform=ax2["M"].transAxes,color="red",fontsize=23)
    ax2["M"].text(0.6,0.04,str(newdfM.R2devmodel2.min()),transform=ax2["M"].transAxes,color="green",fontsize=23)
    #ax2["M"].text(0.8,0.05,str(newdfM.R2devmodel4.min()),transform=ax2["M"].transAxes,color="blue",fontsize=23)
    ax2["M"].text(0.8,0.04,str(newdfM.R2devmodel5.min()),transform=ax2["M"].transAxes,color="black",fontsize=23)
    #ax2["M"].tick_params(axis='both',labelsize=20)

    ax2["N"].plot(newdfN.Visitantes/newdfN.Visitantes.max(),newdfN.yhatmodel1/newdfN.Visitantes.max(),"o",color="red",markersize=10)
    ax2["N"].plot(newdfN.Visitantes/newdfN.Visitantes.max(),newdfN.yhatmodel2/newdfN.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["N"].plot(newdfN.Visitantes/newdfN.Visitantes.max(),newdfN.yhatmodel4/newdfN.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["N"].plot(newdfN.Visitantes/newdfN.Visitantes.max(),newdfN.yhatmodel5/newdfN.Visitantes.max(),"o",color="black",markersize=10)
    ax2["N"].plot(np.linspace(min(newdfN.Visitantes)/newdfN.Visitantes.max(),max(newdfN.Visitantes)/newdfN.Visitantes.max(),100),
      np.linspace(min(newdfN.Visitantes)/newdfN.Visitantes.max(),max(newdfN.Visitantes)/newdfN.Visitantes.max(),100),lw=5)
    ax2["N"].set_title(newdfN.IdOAPN.unique()[0],fontsize=20)
    ax2["N"].text(0.4,0.04,str(newdfN.R2devmodel1.min()),transform=ax2["N"].transAxes,color="red",fontsize=23)
    ax2["N"].text(0.6,0.04,str(newdfN.R2devmodel2.min()),transform=ax2["N"].transAxes,color="green",fontsize=23)
    #ax2["N"].text(0.8,0.05,str(newdfN.R2devmodel4.min()),transform=ax2["N"].transAxes,color="blue",fontsize=23)
    ax2["N"].text(0.8,0.04,str(newdfN.R2devmodel5.min()),transform=ax2["N"].transAxes,color="black",fontsize=23)
    #ax2["N"].tick_params(axis='y',labelsize=0)
    #ax2["N"].tick_params(axis='x',labelsize=20)    

    ax2["O"].plot(newdfO.Visitantes/newdfO.Visitantes.max(),newdfO.yhatmodel1/newdfO.Visitantes.max(),"o",color="red",markersize=10)
    ax2["O"].plot(newdfO.Visitantes/newdfO.Visitantes.max(),newdfO.yhatmodel2/newdfO.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["O"].plot(newdfO.Visitantes/newdfO.Visitantes.max(),newdfO.yhatmodel4/newdfO.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["O"].plot(newdfO.Visitantes/newdfO.Visitantes.max(),newdfO.yhatmodel5/newdfO.Visitantes.max(),"o",color="black",markersize=10)
    ax2["O"].plot(np.linspace(min(newdfO.Visitantes)/newdfO.Visitantes.max(),max(newdfO.Visitantes)/newdfO.Visitantes.max(),100),
      np.linspace(min(newdfO.Visitantes)/newdfO.Visitantes.max(),max(newdfO.Visitantes)/newdfO.Visitantes.max(),100),lw=5)
    ax2["O"].set_title(newdfO.IdOAPN.unique()[0],fontsize=20)
    ax2["O"].text(0.4,0.04,str(newdfO.R2devmodel1.min()),transform=ax2["O"].transAxes,color="red",fontsize=23)
    ax2["O"].text(0.6,0.04,str(newdfO.R2devmodel2.min()),transform=ax2["O"].transAxes,color="green",fontsize=23)
    #ax2["O"].text(0.8,0.05,str(newdfO.R2devmodel4.min()),transform=ax2["O"].transAxes,color="blue",fontsize=23)
    ax2["O"].text(0.8,0.04,str(newdfO.R2devmodel5.min()),transform=ax2["O"].transAxes,color="black",fontsize=23)
    #ax2["O"].tick_params(axis='both',labelsize=20)
    #ax2["O"].tick_params(axis='y',labelsize=0)
    #ax2["O"].tick_params(axis='x',labelsize=20) 

    ax2["P"].plot(newdfP.Visitantes/newdfP.Visitantes.max(),newdfP.yhatmodel1/newdfP.Visitantes.max(),"o",color="red",markersize=10)
    ax2["P"].plot(newdfP.Visitantes/newdfP.Visitantes.max(),newdfP.yhatmodel2/newdfP.Visitantes.max(),"o",color="green",markersize=10)
    #ax2["P"].plot(newdfP.Visitantes/newdfP.Visitantes.max(),newdfP.yhatmodel4/newdfP.Visitantes.max(),"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["P"].plot(newdfP.Visitantes/newdfP.Visitantes.max(),newdfP.yhatmodel5/newdfP.Visitantes.max(),"o",color="black",markersize=10)
    ax2["P"].plot(np.linspace(min(newdfP.Visitantes)/newdfP.Visitantes.max(),max(newdfP.Visitantes)/newdfP.Visitantes.max(),100),
      np.linspace(min(newdfP.Visitantes)/newdfP.Visitantes.max(),max(newdfP.Visitantes)/newdfP.Visitantes.max(),100),lw=5)
    ax2["P"].set_title(newdfP.IdOAPN.unique()[0],fontsize=20)
    ax2["P"].text(0.4,0.04,str(newdfP.R2devmodel1.min()),transform=ax2["P"].transAxes,color="red",fontsize=23)
    ax2["P"].text(0.6,0.04,str(newdfP.R2devmodel2.min()),transform=ax2["P"].transAxes,color="green",fontsize=23)
    #ax2["P"].text(0.8,0.05,str(newdfP.R2devmodel4.min()),transform=ax2["P"].transAxes,color="blue",fontsize=23)
    ax2["P"].text(0.8,0.04,str(newdfP.R2devmodel5.min()),transform=ax2["P"].transAxes,color="black",fontsize=23)
    ax2["P"].tick_params(axis='both',labelsize=20)   
    ax2["P"].set_ylim(0.3,1.2)
   
   

    fig2.legend(loc="upper center", ncols=5, fontsize=25,mode="expand")

    for ax in ax2.values():
   
        #ax.xaxis.set_major_formatter(FuncFormatter(scientific_notation))
        #ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation))
        #ax.set_xticks([ 0.25, 0.5,0.75])
        #ax.set_yticks([ 0.25, 0.5,0.75])
        ax.tick_params(axis="both",which="both",labelsize=15)
        ax.grid()
        #ax.set_xlim(0,1.2)
        #ax.set_ylim(0,1.2)


     
   
   


    plt.show()

