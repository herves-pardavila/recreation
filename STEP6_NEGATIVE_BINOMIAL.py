import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg

if __name__== "__main__":

    #prepare the data
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation.csv")
    df.Date=df.Date.astype("category")
    df.Month=df.Month.astype("category")
    df.Year=df.Year.astype("category")
    df.SITE_NAME=df.SITE_NAME.astype("category")
    df.Season=df.Season.astype("category")
    df.Visitantes=df.Visitantes.astype(int)
    df.loc[df.PUD.isna(),"PUD"]=0
    df.loc[df.turistas_total.isna(),"turistas_total"]=0
    df.loc[df.turistas_corregido.isna(),"turistas_corregido"]=0
    
    #df.PUD=df.PUD.astype(int)
    #df.turistas_total=df.turistas_total.astype(int)
    df["logPUD"]=np.log(df.PUD+1)
    df["log_turistas"]=np.log(df.turistas_total+1)
    df["log_turistas_corregido"]=np.log(df.turistas_corregido+1)
    df["SummerPUD"]=df["Summer*PUD"]
    df["logSummerPUD"]=np.log(df["Summer*PUD"]+1)
    df["Summerturistas"]=df["Summer*turistas"]
    df["Summerturistascorregido"]=df["Summer*turistas_corregido"]
    df.info()
    print(df[["logPUD","turistas","turistas_corregido"]].describe())

    #divide between training set and test set
    mask=np.random.rand(len(df))<0.8
    df_train=df[mask]
    #df_train=df
    df_test=df[~mask]


    #poisson model
    expr1="""Visitantes ~ logPUD + SITE_NAME + Season """
    expr2="""Visitantes ~ turistas_corregido + SITE_NAME + Season  + Summerturistascorregido"""
    expr=expr2
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_results.summary())


    #auxiliary regression model
    df_train['BB_LAMBDA'] = poisson_training_results.mu
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Visitantes'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
    ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""
    aux_olsr_results = smf.ols(ols_expr, df_train).fit()
    print(aux_olsr_results.summary())


    #negative_binomial regression
    print("=========================Negative Binomial Regression===================== ")
    nb2_training_results = sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
    print(nb2_training_results.summary())
    print("AIC=",nb2_training_results.aic)
    nb2_predictions = nb2_training_results.get_prediction(X_test)
    predictions_summary_frame = nb2_predictions.summary_frame()
    dftest=pd.DataFrame(X_test).copy()
    dftest["y"]=y_test["Visitantes"]
    dftest["yhat"]=predicted_counts=predictions_summary_frame['mean']
    dftest.reset_index(inplace=True)
    dftest["SITE_NAME"]=pd.from_dummies(dftest[[col for col in X_test if col.startswith("SITE_NAME")]],default_category="SITE_NAME[T.Aiguestortes]")
    dftest["Month"]=pd.from_dummies(dftest[[col for col in X_test if col.startswith("Month")]],default_category="Month[T.1]")
    dftest["Season"]=pd.from_dummies(dftest[[col for col in X_test if col.startswith("Season")]],default_category="Season[T.Fall]")
    print(dftest.info())
    #influence plots
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    smg.regressionplots.influence_plot(nb2_training_results,external=False,ax=ax1)
    ax1.hlines(y=[-2]*100,xmin=0,xmax=5)
    ax1.hlines(y=[2]*100,xmin=0,xmax=5)
    ax1.set_xlim(0,5)
    ax1.set_ylim(-4,5)
    plt.show()

    #plotting
    newdf=dftest[["Intercept","SITE_NAME","Month","Season","y","yhat"]]
    print(newdf)
    newdf=newdf.groupby(by=["SITE_NAME","Season"],as_index=False).sum(numeric_only=True)
    
    fig2 = plt.figure()
    fig2.suptitle('Predicted versus actual visitors')
    ax2=fig2.add_subplot(111)
    ax2.loglog(newdf.y,newdf.yhat,"*")
    ax2.loglog(np.arange(1e2,8e7,10),np.arange(1e2,8e7,10),label="1-1 line")
    ax2.set_ylabel("Predicted visitors")
    ax2.set_ylabel("Actual visitors")
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(newdf.y,newdf.yhat,newdf["SITE_NAME"])]
    plt.show()



                               
                               
