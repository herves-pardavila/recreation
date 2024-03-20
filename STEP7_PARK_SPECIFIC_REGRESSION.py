import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
import time

def ajuste(df,expr,name):
    
    newdf=pd.DataFrame()
    for park in df.SITE_NAME.unique():
        #print("===========================================")
        #print("Park=",park)
        #print("===========================================")
        subdf=df[df.SITE_NAME==park]

        #divide between training set and test set
        mask=np.random.rand(len(subdf))<0.8
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
            dftest=pd.DataFrame(X_test).copy()
            dftest["y"]=y_test["Visitantes"]
            dftest[name]=predicted_counts=predictions_summary_frame['mean']
            dftest.reset_index(inplace=True)
            #dftest["Month"]=pd.from_dummies(dftest[[col for col in X_test if col.startswith("Month")]],default_category="Month[T.1]")
            dftest["Season"]=pd.from_dummies(dftest[[col for col in X_test if col.startswith("Season")]],default_category="Season[T.Fall]")
            dftest["Park"]=park
            #print(dftest)
            newdf=pd.concat([newdf,dftest])

        except ValueError:
                continue
    return newdf
if __name__== "__main__":

    #prepare the data
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation.csv")
    df.Date=df.Date.astype("category")
    df.Month=df.Month.astype("category")
    df.SITE_NAME=df.SITE_NAME.astype("category")
    df.Season=df.Season.astype("category")
    df.Visitantes=df.Visitantes.astype(int)
    df.loc[df.PUD.isna(),"PUD"]=0
    df.loc[df.turistas_total.isna(),"turistas_total"]=0
    df.loc[df.turistas_corregido.isna(),"turistas_corregido"]=0
    df.PUD=df.PUD.astype(int)
    df.turistas_total=df.turistas_total.astype(int)
    df["logPUD"]=np.log(df.PUD+1)
    df["log_turistas"]=np.log(df.turistas_total+1)
    df["log_turistas_corregido"]=np.log(df.turistas_corregido+1)
    df["SummerPUD"]=df["Summer*PUD"]
    df["logSummerPUD"]=np.log(df["Summer*PUD"]+1)
    df["Summerturistas"]=df["Summer*turistas"]
    df["Summerturistascorregido"]=df["Summer*turistas_corregido"]
    df.info()
    print(df.describe())



    #models
    expr1="""Visitantes ~ logPUD  + Season"""
    expr2="""Visitantes ~ turistas_corregido + Season + Summerturistascorregido"""
    expression=expr1
    name="yhatPUD"
    newdf1=ajuste(df,expression,name)
    print(newdf1)

    expression=expr2
    name="yhatINE"
    newdf2=ajuste(df,expression,name)
    print(newdf2)

    # newdfA=newdf[newdf.Park=="Aigüestortes i Estany de Sant Maurici"]
    # newdfB=newdf[newdf.Park=="Archipiélago de Cabrera"]
    # newdfC=newdf[newdf.Park=="Cabañeros"]
    # newdfD=newdf[newdf.Park=="Caldera de Taburiente"]
    # newdfE=newdf[newdf.Park=="Doñana"]
    # newdfF=newdf[newdf.Park=="Garajonay"]
    # newdfG=newdf[newdf.Park=="Islas Atlánticas de Galicia"]
    # newdfH=newdf[newdf.Park=="Monfragüe"]
    # newdfI=newdf[newdf.Park=="Ordesa y Monte Perdido"]
    # newdfJ=newdf[newdf.Park=="Picos de Europa"]
    # newdfK=newdf[newdf.Park=="Sierra Nevada"]
    # newdfL=newdf[newdf.Park=="Sierra de Guadarrama"]
    # newdfM=newdf[newdf.Park=="Sierra de las Nieves"]
    # newdfN=newdf[newdf.Park=="Tablas de Daimiel"]
    # newdfO=newdf[newdf.Park=="Teide National Park"]
    # newdfP=newdf[newdf.Park=="Timanfaya"]

    # fig2=plt.figure()
    # ax2=fig2.subplot_mosaic("""ABCD
    #                      EFGH
    #                      IJKL
    #                      MNOP""")

    # fig2.subplots_adjust(hspace=0.3)

    # ax2["A"].loglog(newdfA.y,newdfA.yhat,"o",color="black")
    # ax2["A"].plot(np.linspace(min(newdfA.y),max(newdfA.y),100),np.linspace(min(newdfA.y),max(newdfA.y),100))
    # ax2["A"].title.set_text(str(newdfA.Park.unique()))
    

    # ax2["B"].loglog(newdfB.y,newdfB.yhat,"o",color="black")
    # ax2["B"].plot(np.linspace(min(newdfB.y),max(newdfB.y),100),np.linspace(min(newdfB.y),max(newdfB.y),100))
    # ax2["B"].title.set_text(str(newdfB.Park.unique()))
    
    # ax2["C"].loglog(newdfC.y,newdfC.yhat,"o",color="black")
    # ax2["C"].plot(np.linspace(min(newdfC.y),max(newdfC.y),100),np.linspace(min(newdfC.y),max(newdfC.y),100))
    # ax2["C"].title.set_text(str(newdfD.Park.unique()))

    # ax2["D"].loglog(newdfD.y,newdfD.yhat,"o",color="black")
    # ax2["D"].plot(np.linspace(min(newdfD.y),max(newdfD.y),100),np.linspace(min(newdfD.y),max(newdfD.y),100))
    # ax2["D"].title.set_text(str(newdfD.Park.unique()))

    # ax2["E"].loglog(newdfE.y,newdfE.yhat,"o",color="black")
    # ax2["E"].plot(np.linspace(min(newdfE.y),max(newdfE.y),100),np.linspace(min(newdfE.y),max(newdfE.y),100))
    # ax2["E"].title.set_text(str(newdfE.Park.unique()))

    # ax2["F"].loglog(newdfF.y,newdfF.yhat,"o",color="black")
    # ax2["F"].plot(np.linspace(min(newdfF.y),max(newdfF.y),100),np.linspace(min(newdfF.y),max(newdfF.y),100))
    # ax2["F"].title.set_text(str(newdfA.Park.unique()))

    # ax2["G"].loglog(newdfG.y,newdfG.yhat,"o",color="black")
    # ax2["G"].plot(np.linspace(min(newdfG.y),max(newdfG.y),100),np.linspace(min(newdfG.y),max(newdfG.y),100))
    # ax2["G"].title.set_text(str(newdfG.Park.unique()))

    # ax2["H"].loglog(newdfH.y,newdfH.yhat,"o",color="black")
    # ax2["H"].plot(np.linspace(min(newdfH.y),max(newdfH.y),100),np.linspace(min(newdfH.y),max(newdfH.y),100))
    # ax2["H"].title.set_text(str(newdfH.Park.unique()))

    # ax2["I"].loglog(newdfI.y,newdfI.yhat,"o",color="black")
    # ax2["I"].plot(np.linspace(min(newdfI.y),max(newdfI.y),100),np.linspace(min(newdfI.y),max(newdfI.y),100))
    # ax2["I"].title.set_text(str(newdfI.Park.unique()))

    # ax2["J"].loglog(newdfJ.y,newdfJ.yhat,"o",color="black")
    # ax2["J"].plot(np.linspace(min(newdfJ.y),max(newdfJ.y),100),np.linspace(min(newdfJ.y),max(newdfJ.y),100))
    # ax2["J"].title.set_text(str(newdfJ.Park.unique()))

    # ax2["K"].loglog(newdfK.y,newdfK.yhat,"o",color="black")
    # ax2["K"].plot(np.linspace(min(newdfK.y),max(newdfK.y),100),np.linspace(min(newdfK.y),max(newdfK.y),100))
    # ax2["K"].title.set_text(str(newdfK.Park.unique()))

    # ax2["L"].loglog(newdfL.y,newdfL.yhat,"o",color="black")
    # ax2["L"].plot(np.linspace(min(newdfL.y),max(newdfL.y),100),np.linspace(min(newdfL.y),max(newdfL.y),100))
    # ax2["L"].title.set_text(str(newdfL.Park.unique()))

    # ax2["M"].loglog(newdfM.y,newdfM.yhat,"o",color="black")
    # ax2["M"].plot(np.linspace(min(newdfM.y),max(newdfM.y),100),np.linspace(min(newdfM.y),max(newdfM.y),100))
    # ax2["M"].title.set_text(str(newdfM.Park.unique()))

    # ax2["N"].loglog(newdfN.y,newdfN.yhat,"o",color="black")
    # ax2["N"].plot(np.linspace(min(newdfN.y),max(newdfN.y),100),np.linspace(min(newdfN.y),max(newdfN.y),100))
    # ax2["N"].title.set_text(str(newdfN.Park.unique()))

    # ax2["O"].loglog(newdfO.y,newdfO.yhat,"o",color="black")
    # ax2["O"].plot(np.linspace(min(newdfO.y),max(newdfO.y),100),np.linspace(min(newdfO.y),max(newdfO.y),100))
    # ax2["O"].title.set_text(str(newdfO.Park.unique()))

    # ax2["P"].loglog(newdfP.y,newdfP.yhat,"o",color="black")
    # ax2["P"].plot(np.linspace(min(newdfP.y),max(newdfP.y),100),np.linspace(min(newdfP.y),max(newdfP.y),100))
    # ax2["P"].title.set_text(str(newdfP.Park.unique()))

    # # ax2["C"].plot(newdfC.Month,newdfC.Visitantes/newdfC.Visitantes.sum(),color="black")
    # # ax2["C"].plot(newdfC.Month,newdfC.turistas_corregido/newdfC.turistas_corregido.sum(),color="red")
    # # ax2["C"].plot(newdfC.Month,newdfC.PUD/newdfC.PUD.sum(),color="blue")
    # # ax2["C"].title.set_text(str(newdfC.SITE_NAME.unique()))
    # # # ax2["C"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    

    # # ax2["D"].plot(newdfD.Month,newdfD.Visitantes/newdfD.Visitantes.sum(),color="black")
    # # ax2["D"].plot(newdfD.Month,newdfD.turistas_corregido/newdfD.turistas_corregido.sum(),color="red")
    # # ax2["D"].plot(newdfD.Month,newdfD.PUD/newdfD.PUD.sum(),color="blue")
    # # ax2["D"].title.set_text(str(newdfD.SITE_NAME.unique()))
    # # # ax2["D"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["E"].plot(newdfE.Month,newdfE.Visitantes/newdfE.Visitantes.sum(),color="black")
    # # ax2["E"].plot(newdfE.Month,newdfE.turistas_corregido/newdfE.turistas_corregido.sum(),color="red")
    # # ax2["E"].plot(newdfE.Month,newdfE.PUD/newdfE.PUD.sum(),color="blue")
    # # ax2["E"].title.set_text(str(newdfE.SITE_NAME.unique()))
    # # # ax2["E"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    
    # # ax2["F"].plot(newdfF.Month,newdfF.Visitantes/newdfF.Visitantes.sum(),color="black")
    # # ax2["F"].plot(newdfF.Month,newdfF.turistas_corregido/newdfF.turistas_corregido.sum(),color="red")
    # # ax2["F"].plot(newdfF.Month,newdfF.PUD/newdfF.PUD.sum(),color="blue")
    # # ax2["F"].title.set_text(str(newdfF.SITE_NAME.unique()))
    # # # ax2["F"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["G"].plot(newdfG.Month,newdfG.Visitantes/newdfG.Visitantes.sum(),color="black")
    # # ax2["G"].plot(newdfG.Month,newdfG.turistas_corregido/newdfG.turistas_corregido.sum(),color="red")
    # # ax2["G"].plot(newdfG.Month,newdfG.PUD/newdfG.PUD.sum(),color="blue")
    # # ax2["G"].title.set_text(str(newdfG.SITE_NAME.unique()))
    # # # ax2["G"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["H"].plot(newdfH.Month,newdfH.Visitantes/newdfH.Visitantes.sum(),color="black")
    # # ax2["H"].plot(newdfH.Month,newdfH.turistas_corregido/newdfH.turistas_corregido.sum(),color="red")
    # # ax2["H"].plot(newdfH.Month,newdfH.PUD/newdfH.PUD.sum(),color="blue")
    # # ax2["H"].title.set_text(str(newdfH.SITE_NAME.unique()))
    # # # ax2["H"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["I"].plot(newdfI.Month,newdfI.Visitantes/newdfI.Visitantes.sum(),color="black")
    # # ax2["I"].plot(newdfI.Month,newdfI.turistas_corregido/newdfI.turistas_corregido.sum(),color="red")
    # # ax2["I"].plot(newdfI.Month,newdfI.PUD/newdfI.PUD.sum(),color="blue")
    # # ax2["I"].title.set_text(str(newdfI.SITE_NAME.unique()))
    # # # ax2["I"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["J"].plot(newdfJ.Month,newdfJ.Visitantes/newdfJ.Visitantes.sum(),color="black")
    # # ax2["J"].plot(newdfJ.Month,newdfJ.turistas_corregido/newdfJ.turistas_corregido.sum(),color="red")
    # # ax2["J"].plot(newdfJ.Month,newdfJ.PUD/newdfJ.PUD.sum(),color="blue")
    # # ax2["J"].title.set_text(str(newdfJ.SITE_NAME.unique()))
    # # # ax2["J"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["K"].plot(newdfK.Month,newdfK.Visitantes/newdfK.Visitantes.sum(),color="black")
    # # ax2["K"].plot(newdfK.Month,newdfK.turistas_corregido/newdfK.turistas_corregido.sum(),color="red")
    # # ax2["K"].plot(newdfK.Month,newdfK.PUD/newdfK.PUD.sum(),color="blue")
    # # ax2["K"].title.set_text(str(newdfK.SITE_NAME.unique()))
    # # # ax2["K"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["L"].plot(newdfL.Month,newdfL.Visitantes/newdfL.Visitantes.sum(),color="black")
    # # ax2["L"].plot(newdfL.Month,newdfL.turistas_corregido/newdfL.turistas_corregido.sum(),color="red")
    # # ax2["L"].plot(newdfL.Month,newdfL.PUD/newdfL.PUD.sum(),color="blue")
    # # ax2["L"].title.set_text(str(newdfL.SITE_NAME.unique()))
    # # # ax2["L"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["M"].plot(newdfM.Month,newdfM.Visitantes/newdfM.Visitantes.sum(),color="black")
    # # ax2["M"].plot(newdfM.Month,newdfM.turistas_corregido/newdfM.turistas_corregido.sum(),color="red")
    # # ax2["M"].plot(newdfM.Month,newdfM.PUD/newdfM.PUD.sum(),color="blue")
    # # ax2["M"].title.set_text(str(newdfM.SITE_NAME.unique()))
    # # # ax2["M"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["N"].plot(newdfN.Month,newdfN.Visitantes/newdfN.Visitantes.sum(),color="black")
    # # ax2["N"].plot(newdfN.Month,newdfN.turistas_corregido/newdfN.turistas_corregido.sum(),color="red")
    # # ax2["N"].plot(newdfN.Month,newdfN.PUD/newdfN.PUD.sum(),color="blue")
    # # ax2["N"].title.set_text(str(newdfN.SITE_NAME.unique()))
    # # # ax2["N"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["O"].plot(newdfO.Month,newdfO.Visitantes/newdfO.Visitantes.sum(),color="black")
    # # ax2["O"].plot(newdfO.Month,newdfO.turistas_corregido/newdfO.turistas_corregido.sum(),color="red")
    # # ax2["O"].plot(newdfO.Month,newdfO.PUD/newdfO.PUD.sum(),color="blue")
    # # ax2["O"].title.set_text(str(newdfO.SITE_NAME.unique()))
    # # # ax2["O"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    # # ax2["P"].plot(newdfP.Month,newdfP.Visitantes/newdfP.Visitantes.sum(),color="black")
    # # ax2["P"].plot(dfmensualP.Month,dfmensualP.turistas_corregido/dfmensualP.turistas_corregido.sum(),color="red")
    # # ax2["P"].plot(dfmensualP.Month,dfmensualP.PUD/dfmensualP.PUD.sum(),color="blue")
    # # ax2["P"].title.set_text(str(dfmensualP.SITE_NAME.unique()))
    # # # ax2["P"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    # plt.show()
