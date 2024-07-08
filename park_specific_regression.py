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
    null_expr="""Visitantes ~ Season + covid """
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
            y_train, X_train = dmatrices(null_expr, df_train, return_type='dataframe')
            nb2_intercept_only= sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= aux_olsr_results.params[0] )).fit()
            df_train["R2dev"+name]=1-(nb2_training_results.deviance/nb2_intercept_only.deviance)
  
            #df_test.loc[df_test.index,"R2yhat"+name]=r2
            #df_test.loc[df_test.index,"corryhat"+name]=df_test[["yhat"+name,name]].corr(method="spearman").loc["yhat"+name][name]
            log_likelihood+=[nb2_training_results.llf]
            newdf=pd.concat([newdf,df_train])

        except ValueError:
                continue
    log_likelihood=np.array(log_likelihood)
    print("Suma de los log-likelihood=",np.sum(log_likelihood))
    newdf=newdf[["Date","IdOAPN","Visitantes"]+["yhat"+name]+["R2dev"+name]]
    newdf=newdf.round({"R2dev"+name:2})
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
    expr1="""Visitantes ~ logPUD + Season + covid"""
    df1=dataframe.dropna(subset=["Visitantes","PUD"])
    newdf1=ajuste(df1,expr1,"model1")
    
    #model2
    expr2="""Visitantes ~ turistas_total+ Season + covid"""
    df2=dataframe.dropna(subset=["Visitantes","turistas_total"])
    df2.turistas_total=df2.turistas_total.astype(int)
    newdf2=ajuste(df2,expr2,"model2")
    
    #model4
    expr4="""Visitantes ~ logPUD + logIUD + Season + covid"""
    df4=dataframe.dropna(subset=["Visitantes","IUD","PUD"])
    df4["logIUD"]=np.log(df4.IUD+1)
    newdf4=ajuste(df4,expr4,"model4")

    #model 5
    expr5="""Visitantes ~ logPUD + turistas_total+ Season + covid"""
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
    ax2=fig2.subplot_mosaic("""ABCD
                         EFGH
                         IJKL
                         MNOP""")
    
    fig2.subplots_adjust(hspace=0.45,left=0.075,right=0.95, top=0.9, bottom=0.075)
    fig2.text(x=0.45,y=0.025,s="Observed Visitors",fontsize=15)
    fig2.text(x=0.025,y=0.45,s="Estimated Visitors",rotation="vertical",fontsize=15)
    
    ax2["A"].loglog(newdfA.Visitantes,newdfA.yhatmodel1,"o",color="pink",label="Model 1")
    ax2["A"].loglog(newdfA.Visitantes,newdfA.yhatmodel2,"o",color="red",label="Model 2")
    ax2["A"].loglog(newdfA.Visitantes,newdfA.yhatmodel4,"o",color="blue",label="Model 4")
    ax2["A"].loglog(newdfA.Visitantes,newdfA.yhatmodel5,"o",color="black",label="Model 5")
    ax2["A"].plot(np.linspace(min(newdfA.Visitantes),max(newdfA.Visitantes),100),np.linspace(min(newdfA.Visitantes),max(newdfA.Visitantes),100),label="1-1 line")
    ax2["A"].set_title(newdfA.IdOAPN.unique()[0],fontsize=12)
    ax2["A"].text(0.8,0.35,str(newdfA.R2devmodel1.min()),transform=ax2["A"].transAxes,color="pink",fontsize=12)
    ax2["A"].text(0.8,0.25,str(newdfA.R2devmodel2.min()),transform=ax2["A"].transAxes,color="red",fontsize=12)
    #ax2["A"].text(0.8,0.15,str(newdfA.R2devmodel4.min()),transform=ax2["A"].transAxes,color="blue",fontsize=12)
    ax2["A"].text(0.8,0.05,str(newdfA.R2devmodel5.min()),transform=ax2["A"].transAxes,color="black",fontsize=12)
    ax2["A"].tick_params(axis='both',labelsize=15)

    ax2["B"].loglog(newdfB.Visitantes,newdfB.yhatmodel1,"o",color="pink")
    ax2["B"].loglog(newdfB.Visitantes,newdfB.yhatmodel2,"o",color="red")
    ax2["B"].loglog(newdfB.Visitantes,newdfB.yhatmodel4,"o",color="blue")
    ax2["B"].loglog(newdfB.Visitantes,newdfB.yhatmodel5,"o",color="black")
    ax2["B"].plot(np.linspace(min(newdfB.Visitantes),max(newdfB.Visitantes),100),np.linspace(min(newdfB.Visitantes),max(newdfB.Visitantes),100))
    ax2["B"].set_title(newdfB.IdOAPN.unique()[0],fontsize=15)
    ax2["B"].text(0.8,0.35,str(newdfB.R2devmodel1.min()),transform=ax2["B"].transAxes,color="pink",fontsize=12)
    ax2["B"].text(0.8,0.25,str(newdfB.R2devmodel2.min()),transform=ax2["B"].transAxes,color="red",fontsize=12)
    #ax2["B"].text(0.8,0.15,str(newdfB.R2devmodel4.min()),transform=ax2["B"].transAxes,color="blue",fontsize=12)
    ax2["B"].text(0.8,0.05,str(newdfB.R2devmodel5.min()),transform=ax2["B"].transAxes,color="black",fontsize=12)
    ax2["B"].tick_params(axis='both',labelsize=15)

    ax2["C"].loglog(newdfC.Visitantes,newdfC.yhatmodel1,"o",color="pink")
    ax2["C"].loglog(newdfC.Visitantes,newdfC.yhatmodel2,"o",color="red")
    ax2["C"].loglog(newdfC.Visitantes,newdfC.yhatmodel4,"o",color="blue")
    ax2["C"].loglog(newdfC.Visitantes,newdfC.yhatmodel5,"o",color="black")
    ax2["C"].plot(np.linspace(min(newdfC.Visitantes),max(newdfC.Visitantes),100),np.linspace(min(newdfC.Visitantes),max(newdfC.Visitantes),100))
    ax2["C"].set_title(newdfC.IdOAPN.unique()[0],fontsize=15)
    ax2["C"].text(0.8,0.35,str(newdfC.R2devmodel1.min()),transform=ax2["C"].transAxes,color="pink",fontsize=12)
    ax2["C"].text(0.8,0.25,str(newdfC.R2devmodel2.min()),transform=ax2["C"].transAxes,color="red",fontsize=12)
    #ax2["C"].text(0.8,0.15,str(newdfC.R2devmodel4.min()),transform=ax2["C"].transAxes,color="blue",fontsize=12)
    ax2["C"].text(0.8,0.05,str(newdfC.R2devmodel5.min()),transform=ax2["C"].transAxes,color="black",fontsize=12)
    ax2["C"].tick_params(axis='both',labelsize=15)


    ax2["D"].loglog(newdfD.Visitantes,newdfD.yhatmodel1,"o",color="pink")
    ax2["D"].loglog(newdfD.Visitantes,newdfD.yhatmodel2,"o",color="red")
    ax2["D"].loglog(newdfD.Visitantes,newdfD.yhatmodel4,"o",color="blue")
    ax2["D"].loglog(newdfD.Visitantes,newdfD.yhatmodel5,"o",color="black")
    ax2["D"].plot(np.linspace(min(newdfD.Visitantes),max(newdfD.Visitantes),100),np.linspace(min(newdfD.Visitantes),max(newdfD.Visitantes),100))
    ax2["D"].set_title(newdfD.IdOAPN.unique()[0],fontsize=15)
    ax2["D"].text(0.8,0.35,str(newdfD.R2devmodel1.min()),transform=ax2["D"].transAxes,color="pink",fontsize=12)
    ax2["D"].text(0.8,0.25,str(newdfD.R2devmodel2.min()),transform=ax2["D"].transAxes,color="red",fontsize=12)
    #ax2["D"].text(0.8,0.15,str(newdfD.R2devmodel4.min()),transform=ax2["D"].transAxes,color="blue",fontsize=12)
    ax2["D"].text(0.8,0.05,str(newdfD.R2devmodel5.min()),transform=ax2["D"].transAxes,color="black",fontsize=12)
    ax2["D"].tick_params(axis='both',labelsize=15)

    ax2["E"].loglog(newdfE.Visitantes,newdfE.yhatmodel1,"o",color="pink")
    ax2["E"].loglog(newdfE.Visitantes,newdfE.yhatmodel2,"o",color="red")
    ax2["E"].loglog(newdfE.Visitantes,newdfE.yhatmodel4,"o",color="blue")
    ax2["E"].loglog(newdfE.Visitantes,newdfE.yhatmodel5,"o",color="black")
    ax2["E"].plot(np.linspace(min(newdfE.Visitantes),max(newdfE.Visitantes),100),np.linspace(min(newdfE.Visitantes),max(newdfE.Visitantes),100))
    ax2["E"].set_title(newdfE.IdOAPN.unique()[0],fontsize=15)
    ax2["E"].text(0.8,0.35,str(newdfE.R2devmodel1.min()),transform=ax2["E"].transAxes,color="pink",fontsize=12)
    ax2["E"].text(0.8,0.25,str(newdfE.R2devmodel2.min()),transform=ax2["E"].transAxes,color="red",fontsize=12)
    #ax2["E"].text(0.8,0.15,str(newdfE.R2devmodel4.min()),transform=ax2["E"].transAxes,color="blue",fontsize=12)
    ax2["E"].text(0.8,0.05,str(newdfE.R2devmodel5.min()),transform=ax2["E"].transAxes,color="black",fontsize=12)
    ax2["E"].tick_params(axis='both',labelsize=15)

    ax2["F"].loglog(newdfF.Visitantes,newdfF.yhatmodel1,"o",color="pink")
    ax2["F"].loglog(newdfF.Visitantes,newdfF.yhatmodel2,"o",color="red")
    ax2["F"].loglog(newdfF.Visitantes,newdfF.yhatmodel4,"o",color="blue")
    ax2["F"].loglog(newdfF.Visitantes,newdfF.yhatmodel5,"o",color="black")
    ax2["F"].plot(np.linspace(min(newdfF.Visitantes),max(newdfF.Visitantes),100),np.linspace(min(newdfF.Visitantes),max(newdfF.Visitantes),100))
    ax2["F"].set_title(newdfF.IdOAPN.unique()[0],fontsize=15)
    ax2["F"].text(0.8,0.35,str(newdfF.R2devmodel1.min()),transform=ax2["F"].transAxes,color="pink",fontsize=12)
    ax2["F"].text(0.8,0.25,str(newdfF.R2devmodel2.min()),transform=ax2["F"].transAxes,color="red",fontsize=12)
    #ax2["F"].text(0.8,0.15,str(newdfF.R2devmodel4.min()),transform=ax2["F"].transAxes,color="blue",fontsize=12)
    ax2["F"].text(0.8,0.05,str(newdfF.R2devmodel5.min()),transform=ax2["F"].transAxes,color="black",fontsize=12)
    ax2["F"].tick_params(axis='both',labelsize=15)

    ax2["G"].loglog(newdfG.Visitantes,newdfG.yhatmodel1,"o",color="pink")
    ax2["G"].loglog(newdfG.Visitantes,newdfG.yhatmodel2,"o",color="red")
    ax2["G"].loglog(newdfG.Visitantes,newdfG.yhatmodel4,"o",color="blue")
    ax2["G"].loglog(newdfG.Visitantes,newdfG.yhatmodel5,"o",color="black")
    ax2["G"].plot(np.linspace(min(newdfG.Visitantes),max(newdfG.Visitantes),100),np.linspace(min(newdfG.Visitantes),max(newdfG.Visitantes),100))
    ax2["G"].set_title(newdfG.IdOAPN.unique()[0],fontsize=15)
    ax2["G"].text(0.8,0.35,str(newdfG.R2devmodel1.min()),transform=ax2["G"].transAxes,color="pink",fontsize=12)
    ax2["G"].text(0.8,0.25,str(newdfG.R2devmodel2.min()),transform=ax2["G"].transAxes,color="red",fontsize=12)
    ax2["G"].text(0.8,0.15,str(newdfG.R2devmodel4.min()),transform=ax2["G"].transAxes,color="blue",fontsize=12)
    ax2["G"].text(0.8,0.05,str(newdfG.R2devmodel5.min()),transform=ax2["G"].transAxes,color="black",fontsize=12)
    ax2["G"].tick_params(axis='both',labelsize=15)

    ax2["H"].loglog(newdfH.Visitantes,newdfH.yhatmodel1,"o",color="pink")
    ax2["H"].loglog(newdfH.Visitantes,newdfH.yhatmodel2,"o",color="red")
    ax2["H"].loglog(newdfH.Visitantes,newdfH.yhatmodel4,"o",color="blue")
    ax2["H"].loglog(newdfH.Visitantes,newdfH.yhatmodel5,"o",color="black")
    ax2["H"].plot(np.linspace(min(newdfH.Visitantes),max(newdfH.Visitantes),100),np.linspace(min(newdfH.Visitantes),max(newdfH.Visitantes),100))
    ax2["H"].set_title(newdfH.IdOAPN.unique()[0],fontsize=15)
    ax2["H"].text(0.8,0.35,str(newdfH.R2devmodel1.min()),transform=ax2["H"].transAxes,color="pink",fontsize=12)
    ax2["H"].text(0.8,0.25,str(newdfH.R2devmodel2.min()),transform=ax2["H"].transAxes,color="red",fontsize=12)
   # ax2["H"].text(0.8,0.15,str(newdfH.R2devmodel4.min()),transform=ax2["H"].transAxes,color="blue",fontsize=12)
    ax2["H"].text(0.8,0.05,str(newdfH.R2devmodel5.min()),transform=ax2["H"].transAxes,color="black",fontsize=12)
    ax2["H"].tick_params(axis='both',labelsize=15)

    ax2["I"].loglog(newdfI.Visitantes,newdfI.yhatmodel1,"o",color="pink")
    ax2["I"].loglog(newdfI.Visitantes,newdfI.yhatmodel2,"o",color="red")
    ax2["I"].loglog(newdfI.Visitantes,newdfI.yhatmodel4,"o",color="blue")
    ax2["I"].loglog(newdfI.Visitantes,newdfI.yhatmodel5,"o",color="black")
    ax2["I"].plot(np.linspace(min(newdfI.Visitantes),max(newdfI.Visitantes),100),np.linspace(min(newdfI.Visitantes),max(newdfI.Visitantes),100))
    ax2["I"].set_title(newdfI.IdOAPN.unique()[0],fontsize=15)
    ax2["I"].text(0.8,0.35,str(newdfI.R2devmodel1.min()),transform=ax2["I"].transAxes,color="pink",fontsize=12)
    ax2["I"].text(0.8,0.25,str(newdfI.R2devmodel2.min()),transform=ax2["I"].transAxes,color="red",fontsize=12)
    #ax2["I"].text(0.8,0.15,str(newdfI.R2devmodel4.min()),transform=ax2["I"].transAxes,color="blue",fontsize=12)
    ax2["I"].text(0.8,0.05,str(newdfI.R2devmodel5.min()),transform=ax2["I"].transAxes,color="black",fontsize=12)
    ax2["I"].tick_params(axis='both',labelsize=15)

    ax2["J"].loglog(newdfJ.Visitantes,newdfJ.yhatmodel1,"o",color="pink")
    ax2["J"].loglog(newdfJ.Visitantes,newdfJ.yhatmodel2,"o",color="red")
    ax2["J"].loglog(newdfJ.Visitantes,newdfJ.yhatmodel4,"o",color="blue")
    ax2["J"].loglog(newdfJ.Visitantes,newdfJ.yhatmodel5,"o",color="black")
    ax2["J"].plot(np.linspace(min(newdfJ.Visitantes),max(newdfJ.Visitantes),100),np.linspace(min(newdfJ.Visitantes),max(newdfJ.Visitantes),100))
    ax2["J"].set_title(newdfJ.IdOAPN.unique()[0],fontsize=15)
    ax2["J"].text(0.8,0.35,str(newdfJ.R2devmodel1.min()),transform=ax2["J"].transAxes,color="pink",fontsize=12)
    ax2["J"].text(0.8,0.25,str(newdfJ.R2devmodel2.min()),transform=ax2["J"].transAxes,color="red",fontsize=12)
    #ax2["J"].text(0.8,0.15,str(newdfJ.R2devmodel4.min()),transform=ax2["J"].transAxes,color="blue",fontsize=12)
    ax2["J"].text(0.8,0.05,str(newdfJ.R2devmodel5.min()),transform=ax2["J"].transAxes,color="black",fontsize=12)
    ax2["J"].tick_params(axis='both',labelsize=15)

    ax2["K"].loglog(newdfK.Visitantes,newdfK.yhatmodel1,"o",color="pink")
    ax2["K"].loglog(newdfK.Visitantes,newdfK.yhatmodel2,"o",color="red")
    ax2["K"].loglog(newdfK.Visitantes,newdfK.yhatmodel4,"o",color="blue")
    ax2["K"].loglog(newdfK.Visitantes,newdfK.yhatmodel5,"o",color="black")
    ax2["K"].plot(np.linspace(min(newdfK.Visitantes),max(newdfK.Visitantes),100),np.linspace(min(newdfK.Visitantes),max(newdfK.Visitantes),100))
    ax2["K"].set_title(newdfK.IdOAPN.unique()[0],fontsize=15)
    ax2["K"].text(0.8,0.35,str(newdfK.R2devmodel1.min()),transform=ax2["K"].transAxes,color="pink",fontsize=12)
    ax2["K"].text(0.8,0.25,str(newdfK.R2devmodel2.min()),transform=ax2["K"].transAxes,color="red",fontsize=12)
    #ax2["K"].text(0.8,0.15,str(newdfK.R2devmodel4.min()),transform=ax2["K"].transAxes,color="blue",fontsize=12)
    ax2["K"].text(0.8,0.05,str(newdfK.R2devmodel5.min()),transform=ax2["K"].transAxes,color="black",fontsize=12)
    ax2["K"].tick_params(axis='both',labelsize=15)

    ax2["L"].loglog(newdfL.Visitantes,newdfL.yhatmodel1,"o",color="pink")
    ax2["L"].loglog(newdfL.Visitantes,newdfL.yhatmodel2,"o",color="red")
    ax2["L"].loglog(newdfL.Visitantes,newdfL.yhatmodel4,"o",color="blue")
    ax2["L"].loglog(newdfL.Visitantes,newdfL.yhatmodel5,"o",color="black")
    ax2["L"].plot(np.linspace(min(newdfL.Visitantes),max(newdfL.Visitantes),100),np.linspace(min(newdfL.Visitantes),max(newdfL.Visitantes),100))
    ax2["L"].set_title(newdfL.IdOAPN.unique()[0],fontsize=15)
    ax2["L"].text(0.8,0.35,str(newdfL.R2devmodel1.min()),transform=ax2["L"].transAxes,color="pink",fontsize=12)
    ax2["L"].text(0.8,0.25,str(newdfL.R2devmodel2.min()),transform=ax2["L"].transAxes,color="red",fontsize=12)
    #ax2["L"].text(0.8,0.15,str(newdfL.R2devmodel4.min()),transform=ax2["L"].transAxes,color="blue",fontsize=12)
    ax2["L"].text(0.8,0.05,str(newdfL.R2devmodel5.min()),transform=ax2["L"].transAxes,color="black",fontsize=12)
    ax2["L"].tick_params(axis='both',labelsize=15)

    ax2["M"].loglog(newdfM.Visitantes,newdfM.yhatmodel1,"o",color="pink")
    ax2["M"].loglog(newdfM.Visitantes,newdfM.yhatmodel2,"o",color="red")
    ax2["M"].loglog(newdfM.Visitantes,newdfM.yhatmodel4,"o",color="blue")
    ax2["M"].loglog(newdfM.Visitantes,newdfM.yhatmodel5,"o",color="black")
    ax2["M"].plot(np.linspace(min(newdfM.Visitantes),max(newdfM.Visitantes),100),np.linspace(min(newdfM.Visitantes),max(newdfM.Visitantes),100))
    ax2["M"].set_title(newdfM.IdOAPN.unique()[0],fontsize=15)
    ax2["M"].text(0.8,0.35,str(newdfM.R2devmodel1.min()),transform=ax2["M"].transAxes,color="pink",fontsize=12)
    ax2["M"].text(0.8,0.25,str(newdfM.R2devmodel2.min()),transform=ax2["M"].transAxes,color="red",fontsize=12)
    #ax2["M"].text(0.8,0.15,str(newdfM.R2devmodel4.min()),transform=ax2["M"].transAxes,color="blue",fontsize=12)
    ax2["M"].text(0.8,0.05,str(newdfM.R2devmodel5.min()),transform=ax2["M"].transAxes,color="black",fontsize=12)
    ax2["M"].tick_params(axis='both',labelsize=15)

    ax2["N"].loglog(newdfN.Visitantes,newdfN.yhatmodel1,"o",color="pink")
    ax2["N"].loglog(newdfN.Visitantes,newdfN.yhatmodel2,"o",color="red")
    ax2["N"].loglog(newdfN.Visitantes,newdfN.yhatmodel4,"o",color="blue")
    ax2["N"].loglog(newdfN.Visitantes,newdfN.yhatmodel5,"o",color="black")
    ax2["N"].plot(np.linspace(min(newdfN.Visitantes),max(newdfN.Visitantes),100),np.linspace(min(newdfN.Visitantes),max(newdfN.Visitantes),100))
    ax2["N"].set_title(newdfN.IdOAPN.unique()[0],fontsize=15)
    ax2["N"].text(0.8,0.35,str(newdfN.R2devmodel1.min()),transform=ax2["N"].transAxes,color="pink",fontsize=12)
    ax2["N"].text(0.8,0.25,str(newdfN.R2devmodel2.min()),transform=ax2["N"].transAxes,color="red",fontsize=12)
    ax2["N"].text(0.8,0.15,str(newdfN.R2devmodel4.min()),transform=ax2["N"].transAxes,color="blue",fontsize=12)
    ax2["N"].text(0.8,0.05,str(newdfN.R2devmodel5.min()),transform=ax2["N"].transAxes,color="black",fontsize=12)
    ax2["N"].tick_params(axis='both',labelsize=15)

    ax2["O"].loglog(newdfO.Visitantes,newdfO.yhatmodel1,"o",color="pink")
    ax2["O"].loglog(newdfO.Visitantes,newdfO.yhatmodel2,"o",color="red")
    ax2["O"].loglog(newdfO.Visitantes,newdfO.yhatmodel4,"o",color="blue")
    ax2["O"].loglog(newdfO.Visitantes,newdfO.yhatmodel5,"o",color="black")
    ax2["O"].plot(np.linspace(min(newdfO.Visitantes),max(newdfO.Visitantes),100),np.linspace(min(newdfO.Visitantes),max(newdfO.Visitantes),100))
    ax2["O"].set_title(newdfO.IdOAPN.unique()[0],fontsize=15)
    ax2["O"].text(0.8,0.35,str(newdfO.R2devmodel1.min()),transform=ax2["O"].transAxes,color="pink",fontsize=12)
    ax2["O"].text(0.8,0.25,str(newdfO.R2devmodel2.min()),transform=ax2["O"].transAxes,color="red",fontsize=12)
    #ax2["O"].text(0.8,0.15,str(newdfO.R2devmodel4.min()),transform=ax2["O"].transAxes,color="blue",fontsize=12)
    ax2["O"].text(0.8,0.05,str(newdfO.R2devmodel5.min()),transform=ax2["O"].transAxes,color="black",fontsize=12)
    ax2["O"].tick_params(axis='both',labelsize=15)

    ax2["P"].loglog(newdfP.Visitantes,newdfP.yhatmodel1,"o",color="pink")
    ax2["P"].loglog(newdfP.Visitantes,newdfP.yhatmodel2,"o",color="red")
    ax2["P"].loglog(newdfP.Visitantes,newdfP.yhatmodel4,"o",color="blue")
    ax2["P"].loglog(newdfP.Visitantes,newdfP.yhatmodel5,"o",color="black")
    ax2["P"].plot(np.linspace(min(newdfP.Visitantes),max(newdfP.Visitantes),100),np.linspace(min(newdfP.Visitantes),max(newdfP.Visitantes),100))
    ax2["P"].set_title(newdfP.IdOAPN.unique()[0],fontsize=15)
    ax2["P"].text(0.8,0.35,str(newdfP.R2devmodel1.min()),transform=ax2["P"].transAxes,color="pink",fontsize=12)
    ax2["P"].text(0.8,0.25,str(newdfP.R2devmodel2.min()),transform=ax2["P"].transAxes,color="red",fontsize=12)
    ax2["P"].text(0.8,0.15,str(newdfP.R2devmodel4.min()),transform=ax2["P"].transAxes,color="blue",fontsize=12)
    ax2["P"].text(0.8,0.05,str(newdfP.R2devmodel5.min()),transform=ax2["P"].transAxes,color="black",fontsize=12)
    ax2["P"].tick_params(axis='both',labelsize=15)
    fig2.legend(loc="upper center", ncols=5, fontsize=15,mode="expand")



    

    plt.show()

    