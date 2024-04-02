import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
import time
from sklearn.metrics import r2_score
def ajuste(df,expr,name):
    log_likelihood=[]
    newdf=pd.DataFrame()
    for park in df.SITE_NAME.unique():
        #print("===========================================")
        #print("Park=",park)
        #print("===========================================")
        subdf=df[df.SITE_NAME==park]

        #divide between training set and test set
        np.random.seed(seed=1)
        mask=np.random.rand(len(subdf))<0.7
        df_train=subdf[mask]
        #df_train=df
        df_test=subdf[~mask]

        y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
        y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
        try:
            poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
            #print(poisson_training_results.summary())


            #auxiliary regression model
            df_train['BB_LAMBDA'] = poisson_training_results.mu
            df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Visitantes'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
            ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""
            aux_olsr_results = smf.ols(ols_expr, df_train).fit()
            #print(aux_olsr_results.summary())


            #negative_binomial regression
            #print("=========================Negative Binomial Regression===================== ")
            nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
            #print(nb2_training_results.summary())
            #print("AIC=",nb2_training_results.aic)
            nb2_predictions = nb2_training_results.get_prediction(X_test)
            predictions_summary_frame = nb2_predictions.summary_frame()
            df_test["yhat"+name]=predicted_counts=predictions_summary_frame['mean']
            r2=r2_score(df_test[name],df_test["yhat"+name])
            df_test.loc[df_test.index,"R2yhat"+name]=r2
            df_test.loc[df_test.index,"corryhat"+name]=df_test[["yhat"+name,name]].corr(method="spearman").loc["yhat"+name][name]
            log_likelihood+=[nb2_training_results.llf]
            newdf=pd.concat([newdf,df_test])

        except ValueError:
                continue
    log_likelihood=np.array(log_likelihood)
    print("Suma de los log-likelihood=",np.sum(log_likelihood))
    return newdf
if __name__== "__main__":

    #prepare the data
    dataframe=pd.read_csv("/home/usuario/Documentos/recreation/recreation_ready.csv")
    dataframe.Date=dataframe.Date.astype("category")
    dataframe.Month=dataframe.Month.astype("category")
    dataframe.Year=dataframe.Year.astype("category")
    dataframe.SITE_NAME=dataframe.SITE_NAME.astype("category")
    dataframe.Season=dataframe.Season.astype("category")
    print(dataframe.info())
    time.sleep(10)


    #models
    expr1="""Visitantes ~ logPUD  + Season + Summer_logPUD"""
    expr2="""Visitantes ~ turistas_total + Season + Summer_turistas_corregido"""
    expression=expr1
    nombre="logPUD"
    df1=dataframe.dropna(subset=["Visitantes","logPUD"])
    newdf1=ajuste(df1,expression,nombre)
    time.sleep(10)
    
    

    expression=expr2
    nombre="turistas_corregido"
    df2=dataframe.dropna(subset=["Visitantes","turistas_corregido"])
    newdf2=ajuste(df2,expression,nombre)
    time.sleep(10)

    # newdf=pd.concat([newdf1,newdf2])
    # print(newdf.info())

    # newdfA=newdf[newdf.SITE_NAME=="Aigüestortes i Estany de Sant Maurici"]
    # newdfB=newdf[newdf.SITE_NAME=="Archipiélago de Cabrera"]
    # newdfC=newdf[newdf.SITE_NAME=="Cabañeros"]
    # newdfD=newdf[newdf.SITE_NAME=="Caldera de Taburiente"]
    # newdfE=newdf[newdf.SITE_NAME=="Doñana"]
    # newdfF=newdf[newdf.SITE_NAME=="Garajonay"]
    # newdfG=newdf[newdf.SITE_NAME=="Islas Atlánticas de Galicia"]
    # newdfH=newdf[newdf.SITE_NAME=="Monfragüe"]
    # newdfI=newdf[newdf.SITE_NAME=="Ordesa y Monte Perdido"]
    # newdfJ=newdf[newdf.SITE_NAME=="Picos de Europa"]
    # newdfK=newdf[newdf.SITE_NAME=="Sierra Nevada"]
    # newdfL=newdf[newdf.SITE_NAME=="Sierra de Guadarrama"]
    # newdfM=newdf[newdf.SITE_NAME=="Sierra de las Nieves"]
    # newdfN=newdf[newdf.SITE_NAME=="Tablas de Daimiel"]
    # newdfO=newdf[newdf.SITE_NAME=="Teide National Park"]
    # newdfP=newdf[newdf.SITE_NAME=="Timanfaya"]

    # fig2=plt.figure()
    # ax2=fig2.subplot_mosaic("""ABCD
    #                      EFGH
    #                      IJKL
    #                      MNOP""")

    # fig2.subplots_adjust(hspace=0.3)
    

    # ax2["A"].loglog(newdfA.Visitantes,newdfA.yhatturistas_corregido,"o",color="black",label="INE")
    # ax2["A"].loglog(newdfA.Visitantes,newdfA.yhatlogPUD,"o",color="red",label="flickr")
    # ax2["A"].plot(np.linspace(min(newdfA.Visitantes),max(newdfA.Visitantes),100),np.linspace(min(newdfA.Visitantes),max(newdfA.Visitantes),100))
    # ax2["A"].title.set_text(newdfA.SITE_NAME.unique()[0])

    # ax2["B"].loglog(newdfB.Visitantes,newdfB.yhatturistas_corregido,"o",color="black")
    # ax2["B"].loglog(newdfB.Visitantes,newdfB.yhatlogPUD,"o",color="red")
    # ax2["B"].plot(np.linspace(min(newdfB.Visitantes),max(newdfB.Visitantes),100),np.linspace(min(newdfB.Visitantes),max(newdfB.Visitantes),100))
    # ax2["B"].title.set_text(newdfB.SITE_NAME.unique()[0])
    
    # ax2["C"].loglog(newdfC.Visitantes,newdfC.yhatturistas_corregido,"o",color="black")
    # ax2["C"].loglog(newdfC.Visitantes,newdfC.yhatlogPUD,"o",color="red")
    # ax2["C"].plot(np.linspace(min(newdfC.Visitantes),max(newdfC.Visitantes),100),np.linspace(min(newdfC.Visitantes),max(newdfC.Visitantes),100))
    # ax2["C"].title.set_text(newdfC.SITE_NAME.unique()[0])
    
    # ax2["D"].loglog(newdfC.Visitantes,newdfC.yhatturistas_corregido,"o",color="black")
    # ax2["D"].loglog(newdfD.Visitantes,newdfD.yhatlogPUD,"o",color="red")
    # ax2["D"].plot(np.linspace(min(newdfD.Visitantes),max(newdfD.Visitantes),100),np.linspace(min(newdfD.Visitantes),max(newdfD.Visitantes),100))
    # ax2["D"].title.set_text(newdfD.SITE_NAME.unique()[0])

    # ax2["E"].loglog(newdfE.Visitantes,newdfE.yhatturistas_corregido,"o",color="black")
    # ax2["E"].loglog(newdfE.Visitantes,newdfE.yhatlogPUD,"o",color="red")
    # ax2["E"].plot(np.linspace(min(newdfE.Visitantes),max(newdfE.Visitantes),100),np.linspace(min(newdfE.Visitantes),max(newdfE.Visitantes),100))
    # ax2["E"].title.set_text(newdfE.SITE_NAME.unique()[0])

    # ax2["F"].loglog(newdfF.Visitantes,newdfF.yhatturistas_corregido,"o",color="black")
    # ax2["F"].loglog(newdfF.Visitantes,newdfF.yhatlogPUD,"o",color="red")
    # ax2["F"].plot(np.linspace(min(newdfF.Visitantes),max(newdfF.Visitantes),100),np.linspace(min(newdfF.Visitantes),max(newdfF.Visitantes),100))
    # ax2["F"].title.set_text(newdfF.SITE_NAME.unique()[0])

    # ax2["G"].loglog(newdfG.Visitantes,newdfG.yhatturistas_corregido,"o",color="black")
    # ax2["G"].loglog(newdfG.Visitantes,newdfG.yhatlogPUD,"o",color="red")
    # ax2["G"].plot(np.linspace(min(newdfG.Visitantes),max(newdfG.Visitantes),100),np.linspace(min(newdfG.Visitantes),max(newdfG.Visitantes),100))
    # ax2["G"].title.set_text(newdfG.SITE_NAME.unique()[0])

    # ax2["H"].loglog(newdfH.Visitantes,newdfH.yhatturistas_corregido,"o",color="black")
    # ax2["H"].loglog(newdfH.Visitantes,newdfH.yhatlogPUD,"o",color="red")
    # ax2["H"].plot(np.linspace(min(newdfH.Visitantes),max(newdfH.Visitantes),100),np.linspace(min(newdfH.Visitantes),max(newdfH.Visitantes),100))
    # ax2["H"].title.set_text(newdfH.SITE_NAME.unique()[0])

    # ax2["I"].loglog(newdfI.Visitantes,newdfI.yhatturistas_corregido,"o",color="black")
    # ax2["I"].loglog(newdfI.Visitantes,newdfI.yhatlogPUD,"o",color="red")
    # ax2["I"].plot(np.linspace(min(newdfI.Visitantes),max(newdfI.Visitantes),100),np.linspace(min(newdfI.Visitantes),max(newdfI.Visitantes),100))
    # ax2["I"].title.set_text(newdfI.SITE_NAME.unique()[0])

    # ax2["J"].loglog(newdfJ.Visitantes,newdfJ.yhatturistas_corregido,"o",color="black")
    # ax2["J"].loglog(newdfJ.Visitantes,newdfJ.yhatlogPUD,"o",color="red")
    # ax2["J"].plot(np.linspace(min(newdfJ.Visitantes),max(newdfJ.Visitantes),100),np.linspace(min(newdfJ.Visitantes),max(newdfJ.Visitantes),100))
    # ax2["J"].title.set_text(newdfJ.SITE_NAME.unique()[0])

    # ax2["K"].loglog(newdfK.Visitantes,newdfK.yhatturistas_corregido,"o",color="black")
    # ax2["K"].loglog(newdfK.Visitantes,newdfK.yhatlogPUD,"o",color="red")
    # ax2["K"].plot(np.linspace(min(newdfK.Visitantes),max(newdfK.Visitantes),100),np.linspace(min(newdfK.Visitantes),max(newdfK.Visitantes),100))
    # ax2["K"].title.set_text(newdfK.SITE_NAME.unique()[0])

    # ax2["L"].loglog(newdfL.Visitantes,newdfL.yhatturistas_corregido,"o",color="black")
    # ax2["L"].loglog(newdfL.Visitantes,newdfL.yhatlogPUD,"o",color="red")
    # ax2["L"].plot(np.linspace(min(newdfL.Visitantes),max(newdfL.Visitantes),100),np.linspace(min(newdfL.Visitantes),max(newdfL.Visitantes),100))
    # ax2["L"].title.set_text(newdfL.SITE_NAME.unique()[0])

    # ax2["M"].loglog(newdfM.Visitantes,newdfM.yhatturistas_corregido,"o",color="black")
    # ax2["M"].loglog(newdfM.Visitantes,newdfM.yhatlogPUD,"o",color="red")
    # ax2["M"].plot(np.linspace(min(newdfM.Visitantes),max(newdfM.Visitantes),100),np.linspace(min(newdfM.Visitantes),max(newdfM.Visitantes),100))
    # ax2["M"].title.set_text(newdfM.SITE_NAME.unique()[0])

    # ax2["N"].loglog(newdfN.Visitantes,newdfN.yhatturistas_corregido,"o",color="black")
    # ax2["N"].loglog(newdfN.Visitantes,newdfN.yhatlogPUD,"o",color="red")
    # ax2["N"].plot(np.linspace(min(newdfN.Visitantes),max(newdfN.Visitantes),100),np.linspace(min(newdfN.Visitantes),max(newdfN.Visitantes),100))
    # ax2["N"].title.set_text(newdfN.SITE_NAME.unique()[0])

    # ax2["O"].loglog(newdfO.Visitantes,newdfO.yhatturistas_corregido,"o",color="black")
    # ax2["O"].loglog(newdfO.Visitantes,newdfO.yhatlogPUD,"o",color="red")
    # ax2["O"].plot(np.linspace(min(newdfO.Visitantes),max(newdfO.Visitantes),100),np.linspace(min(newdfO.Visitantes),max(newdfO.Visitantes),100))
    # ax2["O"].title.set_text(newdfO.SITE_NAME.unique()[0])

    # ax2["P"].loglog(newdfP.Visitantes,newdfP.yhatturistas_corregido,"o",color="black")
    # ax2["P"].loglog(newdfP.Visitantes,newdfP.yhatlogPUD,"o",color="red")
    # ax2["P"].plot(np.linspace(min(newdfP.Visitantes),max(newdfP.Visitantes),100),np.linspace(min(newdfP.Visitantes),max(newdfP.Visitantes),100))
    # ax2["P"].title.set_text(newdfP.SITE_NAME.unique()[0])

    # fig2.legend()
    # plt.show()
