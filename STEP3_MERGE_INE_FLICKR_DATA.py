import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":

    #datos fotos de flickr
    pud=pd.read_csv("/home/usuario/Documentos/recreation/PCCMMG/pud.csv")
    pud.Date=pd.to_datetime(pud.Date)
    pud.Date=pud.Date.dt.to_period("M")
    pud=pud.groupby(["CODIGOINE","Date"],as_index=False).sum(numeric_only=True)
    print(pud)

    turismo=pd.read_csv("/home/usuario/Documentos/recreation/PCCMMG/turismo.csv")
    turismo["Date"]=pd.to_datetime(turismo.mes)
    turismo.Date=turismo.Date.dt.to_period("M")
    turismo.new_codes=turismo.new_codes.astype(int)
    
    turismo.drop(columns=["dest_cod","mes","COD_PROV","PROVINCIA","NOMBRE_ACTUAL","PERIMETRO"],inplace=True)
    print(turismo)

    #merge tourism data with pud data of flickr
    df=pd.merge(turismo,pud[["Date","CODIGOINE","views","PUD"]],left_on=["Date","new_codes"],right_on=["Date","CODIGOINE"],how="outer")
    df=df[df.new_codes.isin(pud.CODIGOINE.unique())]
    df["indicador_PUD"]=100*df.PUD/df.POBLACION_MUNI
    print(df)
    print(df.info())

    
    #fix some data of this df
    df.loc[df.turistas.isna(),"turistas"]=0.
    df.loc[df.turistas_extranjeros.isna(),"turistas_extranjeros"]=0.
    df.turistas_total=df.turistas+df.turistas_extranjeros
    df["indicador_turismo"]=df.turistas_total/df.POBLACION_MUNI
    df.loc[df.turistas_total==0,"turistas_total"]=np.nan
    
    

    
    df.to_csv("/home/usuario/Documentos/recreation/PCCMMG/recreation_INE_flickr.csv",index=False)

    #plotting
    df=df.groupby(by="new_codes",as_index=False).mean(numeric_only=True)
    geometrias=gpd.read_file("/home/usuario/OneDrive/geo_data/Concellos/Concellos_IGN.shp")
    geometrias.CODIGOINE=geometrias.CODIGOINE.astype(float)
    geometrias=pd.merge(geometrias,df,left_on="CODIGOINE",right_on="new_codes",how="inner")
    
    fig=plt.figure()
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    geometrias.plot(ax=ax1,column="indicador_turismo",legend=True)
    geometrias.plot(ax=ax2,column="indicador_PUD",legend=True)
    plt.show()

    


