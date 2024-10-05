import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt
import pypopulation
if __name__ == "__main__":

    #visitor origins, given by park authority
    df=pd.read_csv("/home/usuario/Documentos/recreation/aiguestortes/procedencias_aiguestortes.csv") #cambiar el nombre del archivo aqui
    #df=df[df.Isla=="Ons"]
    print(df.Lugar.unique())
    
    # #Para las islas atlánticas de Galicia hay que agrupar los datos de las 3 provincias gallegas (no sale Pontevedra)
    # df_galicia=df[df.Zona=="Galicia"]
    # df_galicia=df_galicia[df.Lugar!="Pontevedra"]
    # df_galicia["Lugar"]="Galicia"
    # df_galicia=df_galicia.groupby(by=["Año","Lugar","Zona","Isla"],as_index=False).sum(numeric_only=True)
    # df_galicia["Zona"]="España"
    # print(df_galicia)

    df_españa=df[df.Zona.isin(["España"])]
    #df_españa=pd.concat([df_galicia,df_españa]) # solo necesario para islas atlanticas de galicia
    df_resto=df[df.Zona.isin(["Europa","Mundo"])]

    #ACORDARSE DE CAMBIAR LAS COORDENADAS DEL DESTINO
    #destino=gpd.GeoSeries([Point(-8.775,42.32)],crs="EPSG:4326") #destino Ons
    destino=gpd.GeoSeries([Point(0.9203,42.5759)],crs="EPSG:4326") #destino Aiguestortes
    destino= destino.to_crs("EPSG:3857")
    compute_distances = lambda x: x.distance(destino)[0]

 
    
    # geomtries of spanish provinces for galician data, SOLO NECESARIO PARA ISLAS ATLANTICAS DE GALICIA
    # gdf=gpd.read_file("/home/usuario/OneDrive/curso_qgis/P05.09/provincias.shp")
    # gdf["centroid"]=gdf.geometry.centroid
    
    # gdf["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf["centroid"])))
    # df_galicia=pd.merge(df_galicia,gdf[["provincia","cd_prov","nut2","centroid","geometry","Income","Población","distance (km)"]],left_on="Lugar",right_on="provincia",how="left")
    # print(df_galicia)

    #geometries of autonomous communities for spanish data
    gdf=gpd.read_file("/home/usuario/OneDrive/geo_data/Concellos/CCAA.shp")
    gdf=gdf.to_crs(destino.crs)
    gdf["centroid"]=gdf.geometry.centroid
    gdf["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf["centroid"])))
    df_españa=pd.merge(df_españa,gdf[["text","centroid","geometry","median_inc","Población","distance (km)"]],left_on="Lugar",right_on="text",how="left")
    print(df_españa)

    #geometry of countries for world data
    gdf=gpd.read_file("/home/usuario/OneDrive/geo_data/shp_mapa_paises_mundo_2014/Mapa_paises_mundo.shp")
    destino=destino.to_crs("EPSG:3857")
    gdf=gdf.to_crs("EPSG:3857")
    gdf["centroid"]=gdf.geometry.centroid
    gdf["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf["centroid"])))
    gdf["median_inc"]=gdf["Median Inc"]*1/1.137
    print(gdf)
    gdf=gdf[["CNTR_ID","PAIS","distance (km)","median_inc","geometry"]]
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
    df=pd.concat([df_españa[["Año","Zona","Lugar","Porcentaje","Numero","median_inc","Población","distance (km)"]],
                  df_resto[["Año","Zona","Lugar","Porcentaje","Numero","median_inc","Población","distance (km)"]]])
    print(df[["Lugar","Numero"]])
    print(df.info())
    #print(df[df.Año==2023])
    #print(df[df.Año==2022])
    print(df[df.Año==2019])

    df.to_csv("data_original.csv",index=False)

    

