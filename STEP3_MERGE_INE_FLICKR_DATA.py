import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    
    main_path="/media/david/EXTERNAL_USB/doctorado/"

    #datos fotos de flickr
    pud=pd.read_csv(main_path+"/recreation/pud.csv")
    pud.Date=pd.to_datetime(pud.Date).dt.to_period("M")
    print(pud.info())
    #datos de turismo del INE
    turismo=pd.read_csv(main_path+"/recreation/turismo.csv")
    turismo["Date"]=pd.to_datetime(turismo.mes)
    print(turismo)

    # recover geometries and add overlay with the natural areas
    municipios=gpd.read_file(main_path+"OneDrive/recreation/INE/data/municipios.shp")
    aoi=gpd.read_file(main_path+"/OneDrive/geo_data/protected_zones/parques_nacionales.shp")
    aoi.to_crs("EPSG:3035",inplace=True)
    overlay=gpd.overlay(municipios,aoi,how="intersection")
    overlay["Natural area (km2)"]=overlay.geometry.area/1e6
    overlay["% Natural area"]=overlay["Natural area (km2)"]/overlay["area munic"]
    overlay.sort_values(by="% Natural area",inplace=True)
    print(overlay.info())
    print(overlay[overlay.SITE_NAME=="Picos de Europa"][["NAMEUNIT","new_codes"]])
    #merge tourism data with obverlay, identifying each municipality with its natural park
    turismo=pd.merge(turismo,overlay[["new_codes","SITE_NAME","Natural area (km2)","% Natural area"]],on="new_codes",how="right")
    print(turismo.info())

    #there are several municipalities for each park, let us group so we have a unique value of INE tourist per park and month
    turismo.loc[turismo.turistas.isna(),"turistas"]=0
    turismo.loc[turismo.turistas_extranjeros.isna(),"turistas_extranjeros"]=0
    turismo["turistas_total"]=turismo.turistas+turismo.turistas_extranjeros
    #add new variable
    turismo["turistas_corregido"]=turismo.turistas_total*turismo["% Natural area"]
    turismo.to_csv(main_path+"/recreation/municipalities_info.csv",index=False)
    #group
    turismo=turismo.groupby(by=["Date","SITE_NAME"],as_index=False).sum(numeric_only=True)
   
    #remove unnecessary columns
    turismo.drop(columns=["dest_cod","COD_PROV","PERIMETRO","% Natural area","Natural area (km2)","ALTITUD","SUPERFICIE","POBLACION_MUNI","new_codes"],inplace=True)
    turismo.Date=turismo.Date.dt.to_period("M")
    print(turismo)

    #add missing data as zeros
    date_idx=pd.date_range("7-2019","01-2024",freq="M").to_period("M") #we create a fully complete date range index
    print(date_idx)
    arrays=[date_idx,turismo.SITE_NAME.unique()] #names of the parks
    index=pd.MultiIndex.from_product(arrays,names=["Date","SITE_NAME"]) #we create a double index: parks and month
    turismo.set_index(["Date","SITE_NAME"],inplace=True) #set date and park columns as index in the df
    turismo=turismo.reindex(index) #peform reindex
    turismo.reset_index(inplace=True) #recover the columns
    print(turismo.info())
    turismo.loc[turismo.turistas_total.isna(),"turistas_total"]=0 #set missing observations as zero
    turismo.loc[turismo.turistas_corregido.isna(),"turistas_corregido"]=0 
    print(turismo)
    
    print(pud)
    #merge tourism data with pud data of flickr
    df=pd.merge(turismo,pud,on=["Date","SITE_NAME"],how="outer")
    print(df)
    df.to_csv(main_path+"recreation/recreation_INE_flickr.csv",index=False)

    


