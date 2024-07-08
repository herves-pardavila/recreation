import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":


    #datos de turismo del INE
    turismo=pd.read_csv("/home/usuario/Documentos/recreation/turismo_with_origins.csv")
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

    #turismo=turismo[turismo.SITE_NAME.isin(["Monfragüe"])]
    #print(turismo[["mes","prov_orig","turistas","pais_orig","turistas_extranjeros","NAMEUNIT"]])
    #print(turismo.NAMEUNIT.unique())

    #there are several municipalities for each park, let us group so we have a unique value of INE tourist per park and month

    #add new variable
    turismo["turistas_corregido"]=turismo.turistas*turismo["% Natural area"]
    turismo["turistas_extranjeros_corregido"]=turismo.turistas_extranjeros*turismo["% Natural area"]
    #turismo.to_csv("/home/usuario/Documentos/recreation/municipalities_info.csv",index=False)
    #group
    turismo=turismo.groupby(by=["Date","SITE_NAME","Origen","dest_cod","NAMEUNIT"],as_index=False).sum(numeric_only=True)
    #remove unnecessary columns
    turismo.drop(columns=["dest_cod","COD_PROV","PERIMETRO","% Natural area","Natural area (km2)","ALTITUD","SUPERFICIE","POBLACION_MUNI","new_codes"],inplace=True)
    turismo.Date=turismo.Date.dt.to_period("M")
    print(turismo)
    print(turismo[turismo.SITE_NAME=="Islas Atlánticas de Galicia"])

    turismo.to_csv("/home/usuario/Documentos/recreation/testchi2.csv",index=False)
   

    


