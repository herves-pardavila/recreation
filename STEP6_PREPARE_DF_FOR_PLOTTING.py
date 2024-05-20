import pandas as pd
import numpy as np
from datetime import datetime
if __name__=="__main__":

    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation_INE_FUD_IUD.csv")
    df.Date=pd.to_datetime(df.Date)
    df["Month"]=df.Date.dt.month
    df["Year"]=df.Date.dt.year
    df.loc[df.PUD.isna(),"PUD"]=0
    
    df["logPUD"]=np.log(df.PUD+1)
    #df[df.IUD.notna(),"logIUD"]=np.log(df[df.IUD.notna()]["IUD"]+1)
    df["log_turistas_total"]=np.log(df.turistas_total+1)
    df["log_turistas_corregido"]=np.log(df.turistas_corregido+1)
    df["Season"]=" "
    df["Summer_logPUD"]=np.nan
    df["Summer_logIUD"]=np.nan
    df["Summer_turistas_total"]=np.nan
    df["Summer_turistas_corregido"]=np.nan
    df.loc[df.logPUD.notna(),"Summer_logPUD"]=0
    #df.loc[df.logIUD.notna(),"Summer_logIUD"]=0
    df.loc[df.turistas_total.notna(),"Summer_turistas_total"]=0
    df.loc[df.turistas_corregido.notna(),"Summer_turistas_corregido"]=0
    df.loc[df.Month.isin([1,2,3]),"Season"]="Winter"
    df.loc[df.Month.isin([3,4,5]),"Season"]="Spring"
    df.loc[df.Month.isin([6,7,8,9]),"Season"]="Summer"
    df.loc[df.Month.isin([10,11,12]),"Season"]="Fall"
    df.loc[df.Season=="Summer","Summer_logPUD"]=df.logPUD
    #df.loc[df.Season=="Summer","Summer_logIUD"]=df.logIUD
    df.loc[df.Season=="Summer","Summer_turistas_total"]=df.turistas_total
    df.loc[df.Season=="Summer","Summer_turistas_corregido"]=df.turistas_corregido

    #add covid
    df["covid"]=0
    df.loc[((df.Date > datetime(2020,2,1)) & (df.Date < datetime(2021,6,1))),"covid"]=1
    print(df.covid.unique())
    df=df[~((df.Date > datetime(2020,2,1)) & (df.Date < datetime(2020,8,1)))]
    print(df.info())
    print(df.covid.unique())
    df.to_csv("/home/usuario/Documentos/recreation/recreation_ready.csv",index=False)