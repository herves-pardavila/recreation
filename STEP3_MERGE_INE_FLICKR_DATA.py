import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":

    #datos fotos de flickr
    pud=pd.read_csv("/home/usuario/Documentos/recreation/pud.csv")
    pud.Date=pd.to_datetime(pud.Date)
    pud.Date=pud.Date.dt.to_period("M")
    pud=pud.groupby(["SITE_NAME","Date"],as_index=False).sum(numeric_only=True)
    print(pud)

    turismo=pd.read_csv("/home/usuario/Documentos/recreation/turismo.csv")
    turismo["Date"]=pd.to_datetime(turismo.mes)
    turismo.Date=turismo.Date.dt.to_period("M")
    turismo["indicador_turismo"]=turismo.turistas_total/turismo.POBLACION_MUNI
    print(turismo)

    # recover geometries and add overlay with the natural areas
    municipios=gpd.read_file("/home/usuario/OneDrive/recreation/INE/data/municipios.shp")
    aoi=gpd.read_file("/home/usuario/OneDrive/geo_data/protected_zones/parques_nacionales.shp")
    aoi.to_crs("EPSG:3035",inplace=True)
    overlay=gpd.overlay(municipios,aoi,how="intersection")
    overlay["Natural area (km2)"]=overlay.geometry.area/1e6
    overlay["% Natural area"]=overlay["Natural area (km2)"]/overlay["area munic"]
    overlay.sort_values(by="% Natural area",inplace=True)


    #merge tourism data with obverlay, identifying each municipality with its natural park
    turismo=pd.merge(turismo,overlay[["new_codes","SITE_NAME","Natural area (km2)","% Natural area"]],on="new_codes",how="right")
    print(turismo.info())
    
    turismo.drop(columns=["dest_cod","mes","COD_PROV","PROVINCIA","NOMBRE_ACTUAL","PERIMETRO"],inplace=True)

    #merge tourism data with pud data of flickr
    df=pd.merge(turismo,pud[["Date","SITE_NAME","views","PUD"]],on=["Date","SITE_NAME"],how="outer")
    df["indicador_PUD"]=100*df.PUD/df.POBLACION_MUNI
    print(df)

    
    #fix some data of this df
    df.loc[df.turistas.isna(),"turistas"]=0.
    df.loc[df.turistas_extranjeros.isna(),"turistas_extranjeros"]=0.
    df.turistas_total=df.turistas+df.turistas_extranjeros
    df.loc[df.PUD.isna(),"PUD"]=0.

    
    df.to_csv("/home/usuario/Documentos/recreation/recreation_INE_flickr.csv",index=False)

    


