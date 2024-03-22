import pandas as pd
import geopandas as gpd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
import time
from sklearn.metrics import r2_score

def ajuste(df,expr,name):
    
    newdf=pd.DataFrame()
    for council in df.NAMEUNIT.unique():
        subdf=df[df.NAMEUNIT==council]

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
            df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['turistas_total'] - x['BB_LAMBDA'])**2 - x['BB_LAMBDA']) / x['BB_LAMBDA'], axis=1)
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
            df_test["yhat"]=predicted_counts=predictions_summary_frame['mean']
            r2=r2_score(df_test.turistas_total,df_test.yhat)
            df_test["R2"]=r2
            df_test["corr"]=df_test[["yhat","turistas_total"]].corr(method="spearman").loc["yhat"]["turistas_total"]
        
            
            
            
            newdf=pd.concat([newdf,df_test])

        except ValueError:
                continue
    return newdf
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
    print(df[["indicador_turismo","indicador_PUD"]].corr())


    #models
    expression="""turistas_total ~ logPUD + POBLACION_MUNI + Season + logSummerPUD """
    name="yhatPUD"
    newdf1=ajuste(df,expression,name)
    print(newdf1.info())


   #plot map
    
    gdf=newdf1.groupby(by="NAMEUNIT",as_index=False).mean(numeric_only=True)
    gdf=gdf[["NAMEUNIT","new_codes","R2","corr"]]
    aoi=gpd.read_file("/home/usuario/OneDrive/geo_data/Concellos/Concellos_IGN.shp")
    aoi.CODIGOINE=aoi.CODIGOINE.astype(float)
    gdf=gdf.merge(aoi[["CODIGOINE","geometry"]],left_on="new_codes",right_on="CODIGOINE",how="left")

    print(gdf)
    gdf=gpd.GeoDataFrame(data=gdf,crs=aoi.crs,geometry=gdf.geometry)
    fig=plt.figure()
    ax1=fig.add_subplot(121)
    ax1.title.set_text("R2")
    ax2=fig.add_subplot(122)
    ax2.title.set_text("spearman correlations")
    gdf.plot(ax=ax1,column="R2",legend=True,vmin=0,vmax=1)
    gdf.plot(ax=ax2,column="corr",legend=True)
    plt.show()

    
