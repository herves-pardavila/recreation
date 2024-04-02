import pandas as pd
import numpy as np
if __name__=="__main__":

    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation.csv")
    df.Date=pd.to_datetime(df.Date)
    df["Month"]=df.Date.dt.month
    df["Year"]=df.Date.dt.year
    df.loc[df.PUD.isna(),"PUD"]=0
    df["logPUD"]=np.log(df.PUD+1)
    df["log_turistas"]=np.log(df.turistas_total+1)
    df["log_turistas_corregido"]=np.log(df.turistas_corregido+1)
    df["Season"]=" "
    df["Summer_logPUD"]=np.nan
    df["Summer_turistas"]=np.nan
    df["Summer_turistas_corregido"]=np.nan
    df.loc[df.logPUD.notna(),"Summer_logPUD"]=0
    df.loc[df.turistas_total.notna(),"Summer_turistas"]=0
    df.loc[df.turistas_corregido.notna(),"Summer_turistas_corregido"]=0
    df.loc[df.Month.isin([1,2,3]),"Season"]="Winter"
    df.loc[df.Month.isin([3,4,5]),"Season"]="Spring"
    df.loc[df.Month.isin([6,7,8,9]),"Season"]="Summer"
    df.loc[df.Month.isin([10,11,12]),"Season"]="Fall"
    df.loc[df.Season=="Summer","Summer_logPUD"]=df.logPUD
    df.loc[df.Season=="Summer","Summer_turistas"]=df.turistas_total
    df.loc[df.Season=="Summer","Summer_turistas_corregido"]=df.turistas_corregido
    
    df=df[df.Year>2014]
    
    print(df)
    df.to_csv("/home/usuario/Documentos/recreation/recreation_ready.csv",index=False)