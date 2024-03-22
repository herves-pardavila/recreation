import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
from sklearn.metrics import r2_score
if __name__== "__main__":

    #prepare the data
    df=pd.read_csv("/home/usuario/Documentos/recreation/PCCMMG/recreation_INE_flickr.csv")
    df.Date=pd.to_datetime(df.Date)
    df["Month"]=df.Date.dt.month
    df.Month=df.Month.astype("category")
    df.NAMEUNIT=df.NAMEUNIT.astype("category")
    df.POBLACION_MUNI=df.POBLACION_MUNI.astype(int)
    df.SUPERFICIE=df.SUPERFICIE.astype(int)

    df.loc[df.PUD.isna(),"PUD"]=0
    df.PUD=df.PUD.astype(int)
    df.loc[df.indicador_PUD.isna(),"indicador_PUD"]=0
    df.turistas_total=df.turistas_total.astype(int)
    df["logPUD"]=np.log(df.PUD+1)
    df["logindicadorPUD"]=np.log(df.indicador_PUD+1)
    df["log_turistas"]=np.log(df.turistas_total+1)

    df["Season"]=" "
    df["Summer*PUD"]=0
    df.loc[df.Month.isin([1,2,3]),"Season"]="Winter"
    df.loc[df.Month.isin([3,4,5]),"Season"]="Spring"
    df.loc[df.Month.isin([6,7,8,9]),"Season"]="Summer"
    df.loc[df.Month.isin([10,11,12]),"Season"]="Fall"
    df.loc[df.Season=="Summer","Summer*PUD"]=np.log(df.PUD+1)
    df.Season=df.Season.astype("category")

    df["SummerPUD"]=df["Summer*PUD"]
    df["logSummerPUD"]=np.log(df["Summer*PUD"]+1)
    df["index"]=df.index
    df.info()
    print(df.describe())
    print(df[["indicador_turismo","indicador_PUD"]].corr().loc["indicador_turismo"]["indicador_PUD"])

    #divide between training set and test set
    mask=np.random.rand(len(df))<0.7
    df_train=df[mask]
    #df_train=df
    df_test=df[~mask]
    print(df_test)

    #poisson model
    expr1="""turistas_total ~ logPUD + POBLACION_MUNI + Season + logSummerPUD """
    expr2="""indicador_turismo ~ indicador_PUD + Season + logSummerPUD"""
    expr=expr1
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    print(poisson_training_results.summary())


    #auxiliary regression model
    df_train['BB_LAMBDA'] = poisson_training_results.mu
    df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['turistas_total'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
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
    df_test["yhat"]=predicted_counts=predictions_summary_frame['mean']
    print(df_test)

    # #influence plots
    # fig1=plt.figure()
    # ax1=fig1.add_subplot(111)
    # smg.regressionplots.influence_plot(nb2_training_results,external=False,ax=ax1)
    # ax1.hlines(y=[-2]*100,xmin=0,xmax=5)
    # ax1.hlines(y=[2]*100,xmin=0,xmax=5)
    # ax1.set_xlim(0,5)
    # ax1.set_ylim(-4,5)
    # plt.show()

    #plotting    
    fig2 = plt.figure()
    fig2.suptitle('Predicted versus actual visitors R²=%f'%r2_score(df_test.turistas_total,df_test.yhat))
    ax2=fig2.add_subplot(111)
    ax2.plot(df_test.turistas_total,df_test.yhat,"*")
    ax2.set_ylabel("Predicted visitors")
    ax2.set_xlabel("Actual visitors")

    fig3 = plt.figure()
    fig3.suptitle('Predicted versus actual visitors R²=%f'%r2_score(df_test.turistas_total,df_test.yhat))
    ax3=fig3.add_subplot(111)
    ax3.plot(df_test.index,df_test.yhat,"-*",label="Predicted by Flickr")
    ax3.plot(df_test.index,df_test.turistas_total,"-*",label="Predicted by INE")
    ax3.set_ylabel("Visitors")
    ax3.set_xlabel("Index")
    fig3.legend()
    
    plt.show()



                               
                               
