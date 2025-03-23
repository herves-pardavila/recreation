import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
from sklearn.metrics import r2_score
from scipy import stats
from datetime import datetime
if __name__== "__main__":
    
    
    main_path="/media/david/EXTERNAL_USB/doctorado/"
    #prepare the data
    df=pd.read_csv(main_path+"recreation/recreation_ready.csv")
    df=df[df.Visitantes != 0]
    df=df[df.IdOAPN.isin(["Timanfaya","Islas Atlánticas de Galicia","Tablas de Daimiel","Monfragüe","Cabañeros","Archipiélago de Cabrera"])]
    #df=df[df.Year.isin([2015,2016,2017,2018])]
    df.Date=df.Date.astype("category")
    df.Month=df.Month.astype("category")
    df.Year=df.Year.astype("category")
    df.IdOAPN=df.IdOAPN.astype("category")
    df.Season=df.Season.astype("category")
    df.covid=df.covid.astype("category")


    print(df)
    print(df.info())
    sum_statistics=df[["Visitantes","PUD","IUD","turistas_total","turistas_corregido"]].describe()
    sum_statistics=sum_statistics.round({"Visitantes":0,"PUD":1,"IUD":1,"turistas_total":0,"turistas_corregido":0})
    print(sum_statistics)
    #sum_statistics.to_csv("/home/usuario/Documentos/recreation/imagenes_paper/summary_statistics.csv",index=True)

    print(df[["Visitantes","PUD","IUD","turistas_total","turistas_corregido"]].corr("spearman"))

    dfmean=df[["IdOAPN","Visitantes","PUD","IUD","turistas_total","turistas_corregido","Month"]].groupby(by=["IdOAPN","Month"],as_index=False).mean()
    dfvar=df[["IdOAPN","Visitantes","Month"]].groupby(by=["IdOAPN","Month"],as_index=False).var()
    dfmean=dfmean.merge(dfvar,on=["IdOAPN","Month"],how="inner")
    dfmean["mu/sigma"]=dfmean.Visitantes_x/dfmean.Visitantes_y
    print(dfmean)
    print(dfmean.describe())
    dfmean=dfmean.groupby(by="IdOAPN",as_index=False).mean(numeric_only=True)
    dfmean.rename(columns={"Visitantes_x":"Visitors (mu)","PUD":"FUD","turistas_total":"MPUD","turistas_corregido":"correctedMPUD"},inplace=True)
    dfmean=dfmean.round({"Visitors (mu)":0,"FUD":1,"IUD":1,"MPUD":0,"correctedMPUD":0,"mu/sigma":4})
    dfmean.sort_values(by="Visitors (mu)",inplace=True,ascending=False)
    print(dfmean)
    #dfmean.to_csv("/home/usuario/Documentos/recreation/imagenes_paper/tabla_overdispersion.csv",index=False)

    

    #divide between training set and test set
    df.dropna(subset=["Visitantes","IUD"],inplace=True)
    df.Visitantes=df.Visitantes.astype(int)
    #df.turistas_total=df.turistas_total.astype(int)
    #df["log_Summer_turistas_corregido"]=np.log(df.Summer_turistas_corregido+1)
    df["logIUD"]=np.log(df.IUD+1)
    np.random.seed(seed=1)
    #mask=np.random.rand(len(df))=1
    #df_train=df[mask]
    df_train=df
    #df_test=df[~mask]


    #poisson model
    expr1="""Visitantes ~ logPUD + IdOAPN + Season + covid"""
    expr2="""Visitantes ~ turistas_total+ IdOAPN + Season + covid"""
    expr3="""Visitantes ~ logIUD + IdOAPN + Season + covid -1 """
    expr4="""Visitantes ~ logPUD + logIUD + IdOAPN + Season + covid"""
    expr5="""Visitantes ~ logPUD + turistas_total+ IdOAPN + Season + covid"""
    null_expr="""Visitantes ~ IdOAPN + Season + covid """
    expr=expr3
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    #y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
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
    nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= aux_olsr_results.params[0] )).fit()
    summary=nb2_training_results.summary()
    print(summary)
    print("AIC=",nb2_training_results.aic)
    summary_as_html=summary.tables[1].as_html()
    summary_as_df=pd.read_html(summary_as_html,header=0,index_col=0)[0]
    summary_as_df.loc["DF","coef"]=int(nb2_training_results.df_resid)
    #summary_as_df.loc["Log-likelihood","coef"]=int(nb2_training_results.llf)
    #summary_as_df.loc["Deviance","coef"]=int(nb2_training_results.deviance)
    summary_as_df.loc["chi2","coef"]=int(nb2_training_results.pearson_chi2)
    #summary_as_df.loc["AIC","coef"]=int(nb2_training_results.aic)
 
    # #average marginal effects
    # betas=nb2_training_results.params
    # Z=X_train.dot(betas)
    # ey=np.exp(Z)
    # print("AME of variable %s ="%str(betas.index[-1]),betas.iloc[-1]*ey.mean())
    # print("AME of variable %s ="%str(betas.index[-2]),betas.iloc[-2]*ey.mean())

    #negative binomial regression with intercept only to compute pseudo R2 using deviance
    y_train, X_train = dmatrices(null_expr, df_train, return_type='dataframe')
    nb2_intercept_only= sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= aux_olsr_results.params[0] )).fit()
    summary_as_df.loc["R2dev","coef"]=1-(nb2_training_results.deviance/nb2_intercept_only.deviance)
    print(summary_as_df)
    #summary_as_df.to_csv("/home/usuario/Documentos/recreation/imagenes_paper/tabla_modelo4.csv")





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

    



                               
                               
