import pandas as pd
import numpy as np
from datetime import datetime
if __name__=="__main__":

    df=pd.read_csv("/home/usuario/Documentos/recreation/testchi2.csv")
    df.Date=pd.to_datetime(df.Date)
    df["Month"]=df.Date.dt.month
    df["Year"]=df.Date.dt.year
    df["Season"]=" "
    df.loc[df.Month.isin([1,2,3]),"Season"]="Winter"
    df.loc[df.Month.isin([3,4,5]),"Season"]="Spring"
    df.loc[df.Month.isin([6,7,8,9]),"Season"]="Summer"
    df.loc[df.Month.isin([10,11,12]),"Season"]="Fall"

    #add covid
    df["covid"]=0
    df.loc[((df.Date > datetime(2020,2,1)) & (df.Date < datetime(2021,6,1))),"covid"]=1
    print(df.covid.unique())
    df=df[~((df.Date > datetime(2020,2,1)) & (df.Date < datetime(2020,8,1)))]
    print(df.info())
    print(df)
    print(df.covid.unique())

    df.turistas=df.turistas+df.turistas_extranjeros
    df.turistas_corregido=df.turistas_corregido+df.turistas_extranjeros_corregido
    df.drop(columns=["turistas_extranjeros","turistas_extranjeros_corregido"],inplace=True)
    df.to_csv("/home/usuario/Documentos/recreation/testchi2_ready.csv",index=False)