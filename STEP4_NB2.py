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

    #test data
    test_df=pd.read_csv("/home/usuario/Documentos/recreation/testchi2_ready.csv")
    
    test_df.Date=test_df.Date.astype("category")
    test_df.Month=test_df.Month.astype("category")
    test_df.Year=test_df.Year.astype("category")
    test_df.SITE_NAME=test_df.SITE_NAME.astype("category")
    test_df.Season=test_df.Season.astype("category")
    test_df.covid=test_df.covid.astype("category")
    test_df.turistas=test_df.turistas.astype(int)
 

    park="Islas Atlánticas de Galicia"
    test_df=test_df[test_df.SITE_NAME.isin([park])]
    year=2023
    
    test_df=test_df[test_df.Year==year]
    print(test_df.Year.unique())
    print(test_df)
 
    test_df["Visitantes"]=1
    test_df.rename(columns={"SITE_NAME":"IdOAPN","turistas":"turistas_total"},inplace=True)

    
    print(test_df.info())
    print(test_df.describe())
    

   
    #train data (no origin info)

    #prepare the data
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation_ready.csv")
    df=df[df.IdOAPN.isin([park])]
    df.Date=df.Date.astype("category")
    df.Month=df.Month.astype("category")
    df.Year=df.Year.astype("category")
    df.IdOAPN=df.IdOAPN.astype("category")
    df.Season=df.Season.astype("category")
    df.covid=df.covid.astype("category")

        
    df.dropna(subset=["Visitantes","turistas_total"],inplace=True)
    df.Visitantes=df.Visitantes.astype(int)
    df.turistas_total=df.turistas_total.astype(int)
    #mask=np.random.rand(len(df))=1
    #df_train=df[mask]
    df_train=df
    #df_test=df[~mask]

    expr2="""Visitantes ~ turistas_total + Season + covid"""

    #null_expr="""Visitantes ~ IdOAPN + Season + covid """
    null_expr="""Visitantes ~ 1 """
    expr=expr2
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, test_df, return_type='dataframe')
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
    
    summary_as_html=summary.tables[1].as_html()
    summary_as_df=pd.read_html(summary_as_html,header=0,index_col=0)[0]
    summary_as_df.loc["DF","coef"]=int(nb2_training_results.df_resid)
    summary_as_df.loc["Log-likelihood","coef"]=int(nb2_training_results.llf)
    summary_as_df.loc["Deviance","coef"]=int(nb2_training_results.deviance)
    summary_as_df.loc["chi2","coef"]=int(nb2_training_results.pearson_chi2)
    summary_as_df.loc["AIC","coef"]=int(nb2_training_results.aic)
    #negative binomial regression with intercept only to compute pseudo R2 using deviance
    y_train, X_train = dmatrices(null_expr, df_train, return_type='dataframe')
    nb2_intercept_only= sm.GLM(y_train, X_train,family=sm.families.NegativeBinomial(alpha= aux_olsr_results.params[0] )).fit()
    summary_as_df.loc["R2dev","coef"]=1-(nb2_training_results.deviance/nb2_intercept_only.deviance)
    print(summary_as_df)


    #predictions
    print(X_test)
    nb2_predictions = nb2_training_results.get_prediction(X_test)
    predictions_summary_frame = nb2_predictions.summary_frame()
    test_df["yhat"]=predictions_summary_frame['mean']
    print(test_df)

    print(test_df.Origen.unique())
    test_df=test_df[test_df.Origen.isin(['Andalucía', 'Aragón', 'Asturias, Principado de',
 'Balears, Illes', 'Canarias', 'Cantabria','Castilla - La Mancha' ,'Castilla y León',
  'Cataluña', 'Comunitat Valenciana', 'Extremadura', 'Galicia', 'Madrid, Comunidad de', 
  'Melilla', 'Murcia, Región de', 'Navarra, Comunidad Foral de' ,'País Vasco', 'Rioja, La',"Ceuta",'Total' ])]

#     test_df=test_df[test_df.Origen.isin(['Alemania', 'Andalucía', 'Austria' ,'Bélgica' ,'Canarias', 'Cataluña',
#  'Dinamarca', 'Filipinas' ,'Finlandia', 'Francia', 'Galicia', 'Irlanda',
#  'Islandia', 'Italia' ,'Madrid, Comunidad de', 'Noruega' ,'Países Bajos',
#  'Polonia', 'Reino Unido', 'República Checa' ,'Suecia' ,'Suiza','Comunitat Valenciana' ,'Estonia', 'Indonesia' ,
#  'Portugal', 'Balears, Illes', 'EE.UU.'])]

    #print(test_df[test_df.Origen=='Total'])
    print(test_df[~test_df.Origen.isin(["Total","Galicia"])]["turistas_total"].sum()/test_df["turistas_total"].sum())
    #print(test_df[test_df.Origen.isin(["Galicia"])]["yhat"].sum())