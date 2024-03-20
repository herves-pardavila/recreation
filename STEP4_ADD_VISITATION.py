import pandas as pd
import numpy as np

if __name__=="__main__":
    
    visitantes=pd.read_csv("/home/usuario/Documentos/recreation/visitantes_parques_naturales_2015_2022_limpio.csv")
    visitantes.Date=pd.to_datetime(visitantes.Date)
    print(visitantes[visitantes.IdOAPN=="Sierra Nevada"])
    print("=========================================================================")
   
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation_INE_flickr.csv")
    df.Date=pd.to_datetime(df.Date)
  
    
    df=df.merge(visitantes,left_on=["SITE_NAME","Date"],right_on=["IdOAPN","Date"],how="left")
    df["turistas_corregido"]=df.turistas_total*df["% Natural area"]
    df=df.groupby(by=["SITE_NAME","Date"],as_index=False).sum(numeric_only=True)
    df=df.drop(columns=["new_codes","POBLACION_MUNI","SUPERFICIE","ALTITUD"])
    df.Date=pd.to_datetime(df.Date)
    df["Month"]=df.Date.dt.month
    df["Year"]=df.Date.dt.year
    print(df)
    df.loc[df.turistas_total==0,"turistas_total"]=np.nan
    df.loc[df.PUD==0,"PUD"]=np.nan
    df["Season"]=" "
    df["Summer*PUD"]=0
    df["Summer*turistas"]=0
    df["Summer*turistas_corregido"]=0
    df.loc[df.Month.isin([1,2,3]),"Season"]="Winter"
    df.loc[df.Month.isin([3,4,5]),"Season"]="Spring"
    df.loc[df.Month.isin([6,7,8,9]),"Season"]="Summer"
    df.loc[df.Month.isin([10,11,12]),"Season"]="Fall"
    df.loc[df.Season=="Summer","Summer*PUD"]=np.log(df.PUD+1)
    df.loc[df.Season=="Summer","Summer*turistas"]=df.turistas
    df.loc[df.Season=="Summer","Summer*turistas_corregido"]=df.turistas_corregido

    
    df=df[df.Date.dt.year>2014]
    print(df.info())
    print(df.groupby(by="Month",as_index=False).sum(numeric_only=True))
    df.to_csv("/home/usuario/Documentos/recreation/recreation.csv",index=False)
    

    