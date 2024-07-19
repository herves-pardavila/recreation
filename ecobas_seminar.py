import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from datetime import datetime
def ajuste(df,expr,name):
    log_likelihood=[]
    newdf=pd.DataFrame()
    null_expr1="""Visitantes ~ 1 """
    null_expr2="""Visitantes ~ Season + covid """
    for park in df.IdOAPN.unique():
        #print("===========================================")
        #print("Park=",park)
        #print("===========================================")
        subdf=df[df.IdOAPN==park]

        #divide between training set and test set
        # np.random.seed(seed=1)
        # mask=np.random.rand(len(subdf))<0.7
        # df_train=subdf[mask]
        df_train=subdf
        # df_test=subdf[~mask]

        y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
        #y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
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
            nb2_predictions = nb2_training_results.get_prediction(X_train)
            predictions_summary_frame = nb2_predictions.summary_frame()
            df_train["yhat"+name]=predicted_counts=predictions_summary_frame['mean']

            #negative binomial regression with intercept only to compute pseudo R2 using deviance
            y_train, X_train = dmatrices(null_expr1, df_train, return_type='dataframe')
            nb2_intercept_only= sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= aux_olsr_results.params[0] )).fit()
            df_train["R2dev1"+name]=1-(nb2_training_results.deviance/nb2_intercept_only.deviance)
            y_train, X_train = dmatrices(null_expr2, df_train, return_type='dataframe')
            nb2_intercept_only= sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= aux_olsr_results.params[0] )).fit()
            df_train["R2dev2"+name]=1-(nb2_training_results.deviance/nb2_intercept_only.deviance)
  
            #df_test.loc[df_test.index,"R2yhat"+name]=r2
            #df_test.loc[df_test.index,"corryhat"+name]=df_test[["yhat"+name,name]].corr(method="spearman").loc["yhat"+name][name]
            log_likelihood+=[nb2_training_results.llf]
            newdf=pd.concat([newdf,df_train])

        except ValueError:
                continue
    log_likelihood=np.array(log_likelihood)
    print("Suma de los log-likelihood=",np.sum(log_likelihood))
    newdf=newdf[["Date","IdOAPN","Visitantes"]+["yhat"+name]+["R2dev1"+name]+["R2dev2"+name]]
    newdf=newdf.round({"R2dev1"+name:2,"R2dev2"+name:2})
    return newdf
if __name__== "__main__":

    #prepare the data
    dataframe=pd.read_csv("/home/usuario/Documentos/recreation/recreation_ready.csv")
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
    
    #model2
    expr2="""Visitantes ~ turistas_total+ Season + covid - 1"""
    df2=dataframe.dropna(subset=["Visitantes","turistas_total"])
    df2.turistas_total=df2.turistas_total.astype(int)
    newdf2=ajuste(df2,expr2,"model2")
    
    #model4
    expr4="""Visitantes ~ logPUD + logIUD + Season + covid - 1"""
    df4=dataframe.dropna(subset=["Visitantes","IUD","PUD"])
    df4["logIUD"]=np.log(df4.IUD+1)
    newdf4=ajuste(df4,expr4,"model4")

    #model 5
    expr5="""Visitantes ~ logPUD + turistas_total+ Season + covid - 1"""
    df5=dataframe.dropna(subset=["Visitantes","turistas_total","PUD"])
    df5.turistas_total=df5.turistas_total.astype(int)
    newdf5=ajuste(df5,expr5,"model5")
   

   
    
    newdf=pd.merge(newdf1,newdf2,on=["Date","IdOAPN","Visitantes"],how="outer")
    newdf=pd.merge(newdf,newdf4,on=["Date","IdOAPN","Visitantes"],how="outer")
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

    fig2=plt.figure()
    ax2=fig2.add_subplot(111)

    #fig2.subplots_adjust(hspace=0.45,left=0.075,right=0.95, top=0.9, bottom=0.075)
    fig2.text(x=0.40,y=0.025,s="Observed Visitors",fontsize=15)
    fig2.text(x=0.025,y=0.40,s="Estimated Visitors",rotation="vertical",fontsize=15)
    newdfA=newdfK
    ax2.loglog(newdfA.Visitantes,newdfA.yhatmodel1,"o",color="pink",label="Flickr")
    ax2.loglog(newdfA.Visitantes,newdfA.yhatmodel2,"o",color="red",label="Phones")
    ax2.loglog(newdfA.Visitantes,newdfA.yhatmodel4,"o",color="blue",label="Flickr+Instagram")
    ax2.loglog(newdfA.Visitantes,newdfA.yhatmodel5,"o",color="black",label="Flickr+Phones")
    ax2.plot(np.linspace(min(newdfA.Visitantes),max(newdfA.Visitantes),100),np.linspace(min(newdfA.Visitantes),max(newdfA.Visitantes),100),label="1-1 line")
    ax2.set_title(newdfA.IdOAPN.unique()[0],fontsize=12)
    ax2.text(0.75,0.35,str(newdfA.R2dev1model1.min()),transform=ax2.transAxes,color="pink",fontsize=15)
    ax2.text(0.75,0.25,str(newdfA.R2dev1model2.min()),transform=ax2.transAxes,color="red",fontsize=15)
    #ax2.text(0.75,0.15,str(newdfA.R2dev1model4.min()),transform=ax2.transAxes,color="blue",fontsize=15)#
    ax2.text(0.75,0.05,str(newdfA.R2dev1model5.min()),transform=ax2.transAxes,color="black",fontsize=15)
    ax2.text(0.85,0.35,"("+str(newdfA.R2dev2model1.min())+")",transform=ax2.transAxes,color="pink",fontsize=15)
    ax2.text(0.85,0.25,"(" + str(newdfA.R2dev2model2.min())+")",transform=ax2.transAxes,color="red",fontsize=15)
    #ax2.text(0.85,0.15,"("+str(newdfA.R2dev2model4.min())+")",transform=ax2.transAxes,color="blue",fontsize=15)#
    ax2.text(0.85,0.05,"("+str(newdfA.R2dev2model5.min())+")",transform=ax2.transAxes,color="black",fontsize=15)
    ax2.tick_params(axis='both',labelsize=15)
    fig2.legend(loc="upper center", ncols=5, fontsize=10,mode="expand")
    plt.show()
    fig2.savefig("/home/usuario/Documentos/recreation/imagenes_paper/"+newdfA.IdOAPN.unique()[0]+".png")

    