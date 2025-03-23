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
import warnings
warnings.filterwarnings("ignore")
plt.close("all")
def ajuste(df,expr,name):
    log_likelihood=[]
    newdf=pd.DataFrame()
    null_expr1="""Visitantes ~ 1"""
    
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
            #pgreenicitons
            nb2_predictions = nb2_training_results.get_prediction(X_train)
            predictions_summary_frame = nb2_predictions.summary_frame()
            df_train["yhat"+name]=predicted_counts=predictions_summary_frame['mean']

            #negative binomial regression with intercept only to compute pseudo R2 using deviance
            y_train, X_train = dmatrices(null_expr1, df_train, return_type='dataframe')
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
    main_path="/media/david/EXTERNAL_USB/doctorado/"
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
    expr1="""Visitantes ~ logPUD + Season + covid """
    df1=dataframe.dropna(subset=["Visitantes","PUD"])
    newdf1=ajuste(df1,expr1,"model1")
    
    #model2
    expr2="""Visitantes ~ turistas_total + Season + covid """
    df2=dataframe.dropna(subset=["Visitantes","turistas_total"])
    df2.turistas_total=df2.turistas_total.astype(int)
    newdf2=ajuste(df2,expr2,"model2")
    
    #model4
    expr4="""Visitantes ~ logPUD + logIUD + Season + covid  """
    df4=dataframe.dropna(subset=["Visitantes","IUD","PUD"])
    df4["logIUD"]=np.log(df4.IUD+1)
    newdf4=ajuste(df4,expr4,"model4")

    #model 5
    expr5="""Visitantes ~ logPUD + turistas_total + Season + covid """
    df5=dataframe.dropna(subset=["Visitantes","turistas_total","PUD"])
    df5.turistas_total=df5.turistas_total.astype(int)
    newdf5=ajuste(df5,expr5,"model5")
   

   
    
    newdf=pd.merge(newdf1,newdf2,on=["Date","IdOAPN","Visitantes"],how="outer")
    newdf=pd.merge(newdf,newdf4,on=["Date","IdOAPN","Visitantes"],how="outer")
    newdf=pd.merge(newdf,newdf5,on=["Date","IdOAPN","Visitantes"],how="outer")
    # print(newdf)

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
    #newdfM=newdf[newdf.IdOAPN=="Sierra de las Nieves"]
    newdfN=newdf[newdf.IdOAPN=="Tablas de Daimiel"]
    newdfO=newdf[newdf.IdOAPN=="Teide National Park"]
    newdfP=newdf[newdf.IdOAPN=="Timanfaya"]

    fig2=plt.figure(figsize=(10,10))
    ax2=fig2.subplot_mosaic("""ABC
                          DEF
                          GHI
                          JKL
                          PNO""")
    
    fig2.subplots_adjust(hspace=0.3,wspace=0.1,left=0.075,right=0.95, top=0.9, bottom=0.075)
    fig2.text(x=0.45,y=0.025,s="Observed Visitors",fontsize=35)
    fig2.text(x=0.025,y=0.45,s="Estimated Visitors",rotation="vertical",fontsize=35)
    
    ax2["A"].loglog(newdfA.Visitantes,newdfA.yhatmodel1,"o",color="red",label="Flickr",markersize=10)
    ax2["A"].loglog(newdfA.Visitantes,newdfA.yhatmodel2,"o",color="green",label="Phones",markersize=10)
    ax2["A"].loglog(newdfA.Visitantes,newdfA.yhatmodel4,"o",color="blue",label="Flickr+Instagram",markersize=10)
    ax2["A"].loglog(newdfA.Visitantes,newdfA.yhatmodel5,"o",color="black",label="Flickr+Phones",markersize=10)
    ax2["A"].plot(np.linspace(min(newdfA.Visitantes),max(newdfA.Visitantes),100),
                  np.linspace(min(newdfA.Visitantes),max(newdfA.Visitantes),100),label="1-1 line",lw=5)
    ax2["A"].set_title(newdfA.IdOAPN.unique()[0],fontsize=20)
    ax2["A"].text(0.6,0.05,str(newdfA.R2devmodel1.min()),transform=ax2["A"].transAxes,color="red",fontsize=23)
    ax2["A"].text(0.7,0.05,str(newdfA.R2devmodel2.min()),transform=ax2["A"].transAxes,color="green",fontsize=23)
    #ax2["A"].text(0.8,0.05,str(newdfA.R2devmodel4.min()),transform=ax2["A"].transAxes,color="blue",fontsize=23)
    ax2["A"].text(0.9,0.05,str(newdfA.R2devmodel5.min()),transform=ax2["A"].transAxes,color="black",fontsize=23)
    ax2["A"].tick_params(axis='both',labelsize=20)
    
    ax2["B"].loglog(newdfB.Visitantes,newdfB.yhatmodel1,"o",color="red",markersize=10)
    ax2["B"].loglog(newdfB.Visitantes,newdfB.yhatmodel2,"o",color="green",markersize=10)
    ax2["B"].loglog(newdfB.Visitantes,newdfB.yhatmodel4,"o",color="blue",markersize=10)
    ax2["B"].loglog(newdfB.Visitantes,newdfB.yhatmodel5,"o",color="black",markersize=10)
    ax2["B"].plot(np.linspace(min(newdfB.Visitantes),max(newdfB.Visitantes),100),
                  np.linspace(min(newdfB.Visitantes),max(newdfB.Visitantes),100),lw=5)
    ax2["B"].set_title(newdfB.IdOAPN.unique()[0],fontsize=20)
    ax2["B"].text(0.6,0.05,str(newdfB.R2devmodel1.min()),transform=ax2["B"].transAxes,color="red",fontsize=23)
    ax2["B"].text(0.7,0.05,str(newdfB.R2devmodel2.min()),transform=ax2["B"].transAxes,color="green",fontsize=23)
    #ax2["B"].text(0.8,0.05,str(newdfB.R2devmodel4.min()),transform=ax2["B"].transAxes,color="blue",fontsize=23)
    ax2["B"].text(0.9,0.05,str(newdfB.R2devmodel5.min()),transform=ax2["B"].transAxes,color="black",fontsize=23)
    ax2["B"].tick_params(axis='both',labelsize=20)
    
    ax2["C"].loglog(newdfC.Visitantes,newdfC.yhatmodel1,"o",color="red",markersize=10)
    ax2["C"].loglog(newdfC.Visitantes,newdfC.yhatmodel2,"o",color="green",markersize=10)
    ax2["C"].loglog(newdfC.Visitantes,newdfC.yhatmodel4,"o",color="blue",markersize=10)
    ax2["C"].loglog(newdfC.Visitantes,newdfC.yhatmodel5,"o",color="black",markersize=10)
    ax2["C"].plot(np.linspace(min(newdfC.Visitantes),max(newdfC.Visitantes),100),
                  np.linspace(min(newdfC.Visitantes),max(newdfC.Visitantes),100),lw=5)
    ax2["C"].set_title(newdfC.IdOAPN.unique()[0],fontsize=20)
    ax2["C"].text(0.6,0.05,str(newdfC.R2devmodel1.min()),transform=ax2["C"].transAxes,color="red",fontsize=23)
    ax2["C"].text(0.7,0.05,str(newdfC.R2devmodel2.min()),transform=ax2["C"].transAxes,color="green",fontsize=23)
    #ax2["C"].text(0.8,0.05,str(newdfC.R2devmodel4.min()),transform=ax2["C"].transAxes,color="blue",fontsize=23)
    ax2["C"].text(0.9,0.05,str(newdfC.R2devmodel5.min()),transform=ax2["C"].transAxes,color="black",fontsize=23)
    ax2["C"].tick_params(axis='both',labelsize=20)
    
    
    ax2["D"].loglog(newdfD.Visitantes,newdfD.yhatmodel1,"o",color="red",markersize=10)
    ax2["D"].loglog(newdfD.Visitantes,newdfD.yhatmodel2,"o",color="green",markersize=10)
    ax2["D"].loglog(newdfD.Visitantes,newdfD.yhatmodel4,"o",color="blue",markersize=10)
    ax2["D"].loglog(newdfD.Visitantes,newdfD.yhatmodel5,"o",color="black",markersize=10)
    ax2["D"].plot(np.linspace(min(newdfD.Visitantes),max(newdfD.Visitantes),100),
                  np.linspace(min(newdfD.Visitantes),max(newdfD.Visitantes),100),lw=5)
    ax2["D"].set_title(newdfD.IdOAPN.unique()[0],fontsize=20)
    ax2["D"].text(0.6,0.05,str(newdfD.R2devmodel1.min()),transform=ax2["D"].transAxes,color="red",fontsize=23)
    ax2["D"].text(0.7,0.05,str(newdfD.R2devmodel2.min()),transform=ax2["D"].transAxes,color="green",fontsize=23)
    #ax2["D"].text(0.8,0.05,str(newdfD.R2devmodel4.min()),transform=ax2["D"].transAxes,color="blue",fontsize=23)
    ax2["D"].text(0.9,0.05,str(newdfD.R2devmodel5.min()),transform=ax2["D"].transAxes,color="black",fontsize=23)
    ax2["D"].tick_params(axis='both',labelsize=20)
    
    

        
    
    
    
    
    
    
    fig2.legend(loc="upper center", ncols=5, fontsize=25,mode="expand")



    

    plt.show()

    