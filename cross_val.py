import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
from sklearn import svm
import pandas as pd
from panelsplit import PanelSplit

if __name__ == "__main__":

    #prepare the data
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation_ready.csv")
    #df=df[df.IdOAPN.isin(["Timanfaya"])]
    df.Date=pd.to_datetime(df.Date)
    df.Month=df.Month.astype("category")
    df.Year=df.Year.astype("category")
    df.IdOAPN=df.IdOAPN.astype("category")
    df.Season=df.Season.astype("category")
    df.covid=df.covid.astype("category")
    df.Visitantes=df.Visitantes.astype(int)
    df=df[["Visitantes","turistas_total","Year","Date","Season","covid","IdOAPN"]]
    df=df.dropna()
    df=pd.get_dummies(data=df,columns=["Season","covid","IdOAPN"])

    print(df)
    print(df.info())
    print(df.describe())
    print(df.columns)

    features=['turistas_total','Season_Spring', 'Season_Summer', 'Season_Winter',
    'covid_1','IdOAPN_Archipiélago de Cabrera', 'IdOAPN_Cabañeros',
       'IdOAPN_Caldera de Taburiente', 'IdOAPN_Doñana', 'IdOAPN_Garajonay',
       'IdOAPN_Islas Atlánticas de Galicia', 'IdOAPN_Monfragüe',
       'IdOAPN_Ordesa y Monte Perdido', 'IdOAPN_Picos de Europa',
       'IdOAPN_Sierra Nevada', 'IdOAPN_Sierra de Guadarrama',
       'IdOAPN_Sierra de las Nieves', 'IdOAPN_Tablas de Daimiel',
       'IdOAPN_Teide National Park', 'IdOAPN_Timanfaya']
    #features=["turistas_total"]
    target="Visitantes"

    validation_df=df[df.Year.isin([2020,2021,2022])]
    param_grid={"kernel":["poly"],"degree":[2],"epsilon":[0.1],"coef0":[0,1,10,100,1000],"C":[1e2,1e3,1e4]}
    panel_split = PanelSplit(validation_df.Date, n_splits=5, gap=0, test_size=5, plot=True)
    grid_search = GridSearchCV(svm.SVR(), param_grid=param_grid, cv = panel_split)
    grid_search.fit(validation_df[features], validation_df[target])
    print('GridSearch results:')
    print(grid_search.cv_results_["params"])
    print(grid_search.cv_results_["mean_test_score"])