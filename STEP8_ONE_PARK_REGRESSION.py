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
    #choose one specific park
    df=df[df.SITE_NAME=="Teide National Park"]
    
    df.Date=df.Date.astype("category")
    df.Month=df.Month.astype("category")
    df.Year=df.Year.astype("category")
    
    df.Season=df.Season.astype("category")
    


    print(df)
    print(df.info())
    print(df[["Visitantes","logPUD","turistas_total","turistas_corregido"]].describe())
    print(df[["Visitantes","logPUD","turistas_total","turistas_corregido"]].corr("spearman"))



    

    #divide between training set and test set
    df.dropna(subset=["Visitantes","PUD"],inplace=True)
    df.Visitantes=df.Visitantes.astype(int)
    df.PUD=df.PUD.astype(int)
    #df.turistas_total=df.turistas_total.astype(int)
    df["log_Summer_turistas_corregido"]=np.log(df.Summer_turistas_corregido+1)
    df["log_Summer_turistas"]=np.log(df.Summer_turistas+1)
    print(df.info())

    df_train=df
    


    #poisson model
    expr1="""Visitantes ~ logPUD + SITE_NAME + Season """
    expr2="""Visitantes ~ turistas_total + Season"""
    expr=expr1
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    #print(poisson_training_results.summary())
    print("Mean mu=",np.mean(poisson_training_results.mu))


    #auxiliary regression model
    df_train['BB_LAMBDA'] = poisson_training_results.mu
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Visitantes'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA -1"""
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    #print(aux_olsr_results.summary())
    print("Value of alpha=",aux_olsr_results.params[0])


    #negative_binomial regression
    print("=========================Negative Binomial Regression===================== ")
    nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
    print(nb2_training_results.summary())
    print("AIC=",nb2_training_results.aic)


    nb2_predictions = nb2_training_results.get_prediction(X_train)
    predictions_summary_frame = nb2_predictions.summary_frame()
    df_train["yhat"]=predicted_counts=predictions_summary_frame['mean']
    res=stats.cramervonmises_2samp(df_train.Visitantes,df_train.yhat)
    print(res)
    print(res.pvalue > 0.05)
    # print(df_test.info())


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
    # ax3.set_xlabel("Index")
    # fig3.legend()






    # newdf=df_test[["SITE_NAME","Month","Season","Visitantes","yhat"]]
    # print(newdf)
    # newdf=newdf.groupby(by=["SITE_NAME"],as_index=False).mean(numeric_only=True)
    
    # fig2 = plt.figure()
    # fig2.suptitle('Predicted versus actual visitors')
    # ax2=fig2.add_subplot(111)
    # ax2.loglog(newdf.Visitantes,newdf.yhat,"*")
    # ax2.loglog(np.arange(1e2,8e7,10),np.arange(1e2,8e7,10),label="1-1 line")
    # ax2.set_ylabel("Predicted visitors")
    # ax2.set_ylabel("Actual visitors")
    # [plt.text(i,j,f"{k}") for (i,j,k) in zip(newdf.Visitantes,newdf.yhat,newdf["SITE_NAME"])]
    # plt.show()



                               
                               
