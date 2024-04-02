import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":

    #datos fotos de flickr
    pud=pd.read_csv("/home/usuario/Documentos/recreation/pud.csv")
    pud.Date=pd.to_datetime(pud.Date)
    print(pud.info())
    #datos de turismo del INE
    turismo=pd.read_csv("/home/usuario/Documentos/recreation/turismo.csv")
    turismo["Date"]=pd.to_datetime(turismo.mes)
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

    #there are several municipalities for each park, let us group so we have a unique value of INE tourist per park and month
    turismo.loc[turismo.turistas.isna(),"turistas"]=0
    turismo.loc[turismo.turistas_extranjeros.isna(),"turistas_extranjeros"]=0
    turismo["turistas_total"]=turismo.turistas+turismo.turistas_extranjeros
    #add new variable
    turismo["turistas_corregido"]=turismo.turistas_total*turismo["% Natural area"]
    #group
    turismo=turismo.groupby(by=["Date","SITE_NAME"],as_index=False).sum(numeric_only=True)
    #remove unnecessary columns
    turismo.drop(columns=["dest_cod","COD_PROV","PERIMETRO","% Natural area","Natural area (km2)","ALTITUD","SUPERFICIE","POBLACION_MUNI","new_codes"],inplace=True)
    print(turismo.info())

    #merge tourism data with pud data of flickr
    df=pd.merge(turismo,pud,on=["Date","SITE_NAME"],how="outer")
    print(df)
    df.to_csv("/home/usuario/Documentos/recreation/recreation_INE_flickr.csv",index=False)

    


