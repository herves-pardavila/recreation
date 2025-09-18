import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt
import pypopulation
if __name__ == "__main__":
    
    path=r"F:/doctorado/"
   

    df_interno=pd.read_excel(path+"recreation/ZonalTravelCost/datos_municipales/exp_tmov_interno_mun_2022.xlsx",
                           sheet_name=None) #origenes por CCAA y paises
    
    df_interno=pd.concat([df_interno["2022-01"],df_interno["2022-02"],
                          df_interno["2022-03"],df_interno["2022-04"],
                          df_interno["2022-05"],df_interno["2022-06"],
                          df_interno["2022-07"],df_interno["2022-08"],
                          df_interno["2022-09"],df_interno["2022-10"],
                          df_interno["2022-11"],df_interno["2022-12"]])
    
    df_Ons_2022=df_interno.loc[df_interno.dest.isin(["Bueu","Sanxenxo"]),["mes","mun_orig_cod","mun_orig","dest_cod","dest","turistas"]]
    
    df_interno=pd.read_excel(path+"recreation/ZonalTravelCost/datos_municipales/exp_tmov_interno_mun_2023.xlsx",
                           sheet_name=None) #origenes por CCAA y paises
    
    df_interno=pd.concat([df_interno["2023-01"],df_interno["2023-02"],
                          df_interno["2023-03"],df_interno["2023-04"],
                          df_interno["2023-05"],df_interno["2023-06"],
                          df_interno["2023-07"],df_interno["2023-08"],
                          df_interno["2023-09"],df_interno["2023-10"],
                          df_interno["2023-11"],df_interno["2023-12"]])
    
    df_Ons_2023=df_interno.loc[df_interno.dest.isin(["Bueu","Sanxenxo"]),["mes","mun_orig_cod","mun_orig","dest_cod","dest","turistas"]]
    
    df_Ons =pd.concat([df_Ons_2022,df_Ons_2023])
    
    df_Ons.mes=pd.to_datetime(df_Ons.mes,format="%Y-%m").dt.to_period("M")
    df_Ons["Año"]=df_Ons.mes.dt.year
    df_Ons["Zona"]= "España"    
    df_Ons.rename(columns={"mun_orig":"Lugar","turistas":"turistasINE"},inplace=True)  
    df_Ons=df_Ons[["mes","Año","Lugar","Zona","mun_orig_cod","turistasINE"]]
    df_Ons=df_Ons.groupby(by=["Lugar","Año","Zona","mun_orig_cod"],as_index=False).sum(numeric_only=True) #convertimos en datos anuales
    
    


    #PARTE GEO
    destino=gpd.GeoSeries([Point(-8.775,42.32)],crs="EPSG:4326") #destino Ons    
    destino= destino.to_crs("EPSG:3857")
    compute_distances = lambda x: x.distance(destino)[0]   
    #geometries of spanish municipalities
    gdf=gpd.read_file(path+"OneDrive/recreation/INE/data/municipios.shp")
    gdf.to_crs("EPSG:3857",inplace=True)
    gdf["centroid"]=gdf.geometry.centroid
    gdf=gdf[gdf.centroid!= None]
    gdf["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf["centroid"])))
   
    
    #UNIR LA PARTE GEO CON LA PARTE DE DATOS DE TURSMIO
   
    newdf=pd.merge(df_Ons,gdf[["NAMEUNIT","new_codes","NOMBRE_ACT",
                                  "POBLACION_","distance (km)"]],
                                  left_on="mun_orig_cod",right_on="new_codes",
                                  how="left")
    newdf[["Lugar","Año","Zona","mun_orig_cod","turistasINE","POBLACION_",
            "distance (km)"]].to_csv(path +"recreation/ZonalTravelCost/datos_municipales/3travel_cost_Ons.csv",
                                                        index=False)
       
    newgdf=pd.merge(df_Ons,gdf[["NAMEUNIT","new_codes","NOMBRE_ACT",
                                  "POBLACION_","distance (km)","geometry"]],
                                  left_on="mun_orig_cod",right_on="new_codes",
                                  how="right")
       
    

    newgdf= gpd.GeoDataFrame(data=newgdf[["Lugar","Año","Zona","mun_orig_cod",
                                          "turistasINE","POBLACION_","distance (km)",
                                          "geometry"]],crs=gdf.crs,geometry=newgdf.geometry)
    
    newgdf.to_file(path +"recreation/ZonalTravelCost/datos_municipales/3travel_cost_Ons.gpkg",
                  driver="GPKG",index=False)
    
   #  variable="turistasINE"
    
   #  fig=plt.figure()
   #  ax=fig.add_subplot(111)
   #  newgdf.plot(column=variable, ax= ax)
   #  plt.show()

   
    
