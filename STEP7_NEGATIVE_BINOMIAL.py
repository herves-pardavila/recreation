import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
from sklearn.metrics import r2_score
from scipy import stats
if __name__== "__main__":

    #prepare the data
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation_ready.csv")
    #df=df[df.IdOAPN.isin(["Timanfaya","Islas Atlánticas de Galicia","Tablas de Daimiel"])]
    df.Date=df.Date.astype("category")
    df.Month=df.Month.astype("category")
    df.Year=df.Year.astype("category")
    df.IdOAPN=df.IdOAPN.astype("category")
    df.Season=df.Season.astype("category")


    print(df)
    print(df.info())
    print(df[["Visitantes","logPUD","turistas_total","turistas_corregido"]].describe())
    print(df[["Visitantes","logPUD","turistas_total","turistas_corregido"]].corr("spearman"))

    dfmean=df[["IdOAPN","Visitantes","Month"]].groupby(by=["IdOAPN","Month"],as_index=False).mean()
    dfvar=df[["IdOAPN","Visitantes","Month"]].groupby(by=["IdOAPN","Month"],as_index=False).var()
    dfmean=dfmean.merge(dfvar,on=["IdOAPN","Month"],how="inner")
    dfmean["mu/sigma"]=dfmean.Visitantes_x/dfmean.Visitantes_y
    print(dfmean)
    print(dfmean.describe())


    

    #divide between training set and test set
    df.dropna(subset=["Visitantes","turistas_total"],inplace=True)
    df.Visitantes=df.Visitantes.astype(int)
    #df.turistas_total=df.turistas_total.astype(int)
    df["log_Summer_turistas_corregido"]=np.log(df.Summer_turistas_corregido+1)
    #df["logIUD"]=np.log(df.IUD+1)
    np.random.seed(seed=1)
    mask=np.random.rand(len(df))<0.99
    df_train=df[mask]
    #df_train=df
    df_test=df[~mask]


    #poisson model
    expr1="""Visitantes ~ logPUD + IdOAPN + Season"""
    expr2="""Visitantes ~ turistas_total+ IdOAPN + Season + log_Summer_turistas_corregido"""
    expr3="""Visitantes ~ logIUD + IdOAPN + Season"""
    null_expr="""Visitantes ~ 1"""
    expr=null_expr
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_results.summary())
    print("AIC=",poisson_training_results.aic)
    #print("Mean mu=",poisson_training_results.mu)


    #auxiliary regression model
    df_train['BB_LAMBDA'] = poisson_training_results.mu
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Visitantes'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA -1"""
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    print(aux_olsr_results.summary())
    print("Value of alpha=",aux_olsr_results.params[0])


    #negative_binomial regression
    print("=========================Negative Binomial Regression===================== ")
    nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= 0.158 )).fit()
    print(nb2_training_results.summary())
    print("AIC=",nb2_training_results.aic)


    #nb2_predictions = nb2_training_results.get_prediction(X_test)
    # print(nb2_predictions)
    #predictions_summary_frame = nb2_predictions.summary_frame()
    # print(predictions_summary_frame)
    #df_test["yhat"]=predictions_summary_frame['mean']
    # res=stats.cramervonmises_2samp(df_test.Visitantes,df_test.yhat)
    # print(res)
    # print(res.pvalue > 0.05)

    # # df_test["chi cuadrado"]=((df_test.Visitantes-df_test.yhat)**2)/df_test.yhat
    # # print(df_test[["Visitantes","yhat","chi cuadrado"]])
    # # print("Para un test set con %f observaciones el estadístico X²=%f" %(len(df_train),df_train["chi cuadrado"].sum()))


    # #influence plots
    # fig1=plt.figure()
    # ax1=fig1.add_subplot(111)
    # smg.regressionplots.influence_plot(nb2_training_results,external=False,ax=ax1)
    # ax1.hlines(y=[-2]*100,xmin=0,xmax=5)
    # ax1.hlines(y=[2]*100,xmin=0,xmax=5)
    # ax1.set_xlim(0,5)
    # ax1.set_ylim(-4,5)
    # plt.show()

    # #plotting

    # fig2 = plt.figure()
    # fig2.suptitle('Predicted versus actual visitors R²=%f'%r2_score(df_test.Visitantes,df_test.yhat))
    # ax2=fig2.add_subplot(111)
    # ax2.plot(df_test.Visitantes,df_test.yhat,"*")
    # ax2.set_ylabel("Predicted visitors")
    # ax2.set_xlabel("Actual visitors")

    # fig3 = plt.figure()
    # fig3.suptitle('Predicted versus actual visitors R²=%f'%r2_score(df_test.Visitantes,df_test.yhat))
    # ax3=fig3.add_subplot(111)
    # ax3.plot(df_test.index,df_test.yhat,"-*",label="Predicted by INE")
    # ax3.plot(df_test.index,df_test.Visitantes,"-*",label="Park visitors")
    # ax3.set_ylabel("Visitors")
    # ax3.set_xlabel("Observation")
    # fig3.legend()






    # newdf=df_test[["IdOAPN","Month","Season","Visitantes","yhat"]]
    # print(newdf)
    # newdf=newdf.groupby(by=["IdOAPN"],as_index=False).mean(numeric_only=True)
    
    # fig2 = plt.figure()
    # fig2.suptitle('Predicted versus actual visitors')
    # ax2=fig2.add_subplot(111)
    # ax2.loglog(newdf.Visitantes,newdf.yhat,"*")
    # ax2.loglog(np.arange(1e2,8e7,10),np.arange(1e2,8e7,10),label="1-1 line")
    # ax2.set_ylabel("Predicted visitors")
    # ax2.set_ylabel("Actual visitors")
    # [plt.text(i,j,f"{k}") for (i,j,k) in zip(newdf.Visitantes,newdf.yhat,newdf["IdOAPN"])]
    #plt.show()



                               
                               
