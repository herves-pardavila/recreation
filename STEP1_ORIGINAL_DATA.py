import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt
import pypopulation
if __name__ == "__main__":
    
    path=r"F:/doctorado/"
    
    #visitor origins, given by park authority
   
    df=pd.read_csv(path+"recreation/cabañeros/procedencias_cabañeros.csv",sep=",",na_values="") 

    print(df.Lugar.unique())


    df_españa=df[df.Zona.isin(["España"])]
    df_resto=df[df.Zona.isin(["Europa","Mundo"])]

    #COORDENADAS DEL DESTINO
    #destino=gpd.GeoSeries([Point(0.9203,42.5759)],crs="EPSG:4326") #destino Aiguestortes
    destino=gpd.GeoSeries([Point(-4.5,39.4)],crs="EPSG:4326") #destino Cabañeros
    destino= destino.to_crs("EPSG:3857")
    compute_distances = lambda x: x.distance(destino)[0]



    #geometries of autonomous communities for spanish data
    gdf=gpd.read_file(path+"/OneDrive/geo_data/Concellos/CCAA.shp",engine="pyogrio")
    gdf=gdf.to_crs(destino.crs)
    gdf["centroid"]=gdf.geometry.centroid
    gdf["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf["centroid"])))
    df_españa=pd.merge(df_españa,gdf[["text","centroid","geometry","RBMPP","Población","distance (km)"]],left_on="Lugar",right_on="text",how="left")
    print(df_españa)

    #geometry of countries for world data
    gdf=gpd.read_file(path+"OneDrive/geo_data/shp_mapa_paises_mundo_2014/Mapa_paises_mundo.shp")
    destino=destino.to_crs("EPSG:3857")
    gdf=gdf.to_crs("EPSG:3857")
    gdf["centroid"]=gdf.geometry.centroid
    gdf["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf["centroid"])))
    gdf["RBMPP"]=gdf["ANNIPC"]*1/1.184 #cambiar de dolares estadounidenses a euros del 2021
    print(gdf)
    gdf=gdf[["CNTR_ID","PAIS","distance (km)","RBMPP","geometry"]]
    df_resto=pd.merge(df_resto,gdf,left_on="Lugar",right_on="PAIS",how="left")
    #add population of countries
    for code in df_resto.CNTR_ID.unique():
        try:
            df_resto.loc[df_resto.CNTR_ID==code,"Población"]=pypopulation.get_population(code)
        except AttributeError: 
            continue
    
    
    
    
    df_resto.loc[df_resto.Lugar=="Reino Unido","Población"]=66834405
    df_resto.loc[df_resto.Lugar=="Grecia","Población"]=10716322
    print(df_resto)


    #concatenate back
    df=pd.concat([df_españa[["Año","Zona","Lugar","Porcentaje","Numero","RBMPP","Población","distance (km)"]],
                  df_resto[["Año","Zona","Lugar","Porcentaje","Numero","RBMPP","Población","distance (km)"]]])
    print(df[["Lugar","Numero"]])
    print(df.info())
    df.dropna(subset="distance (km)",inplace=True)
    #print(df[df.Año==2023])
    #print(df[df.Año==2022])
    #print(df[df.Año==2019])

    df.to_csv(path+"recreation/ZonalTravelCost/data_original_Cabañeros.csv",index=False)

    

