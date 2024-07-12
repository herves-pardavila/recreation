import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt
import pypopulation
if __name__ == "__main__":

    #visitor origins, given by park authority
    df=pd.read_csv("/home/usuario/Documentos/recreation/turismo_with_origins.csv")
    df=df[df.NAMEUNIT.isin(["Bueu","Portonovo","Sanxenxo"])]
    print(df)
    df.mes=pd.to_datetime(df.mes,format="%Y-%m").dt.to_period("M")
    df["Año"]=df.mes.dt.year
    df=df[df.Año==2023]
    df["Numero"]=np.nansum([df.turistas,df.turistas_extranjeros],axis=0)
    df.loc[pd.isna(df.turistas_extranjeros),"Zona"]="España"
    df.loc[~pd.isna(df.turistas_extranjeros),"Zona"]="Mundo"
    df.rename(columns={"Origen":"Lugar"},inplace=True)  
    df=df[["mes","Año","Lugar","Zona","Numero"]]
    df=df.groupby(by=["Lugar","Año","Zona"],as_index=False).sum(numeric_only=True)
    df.replace({"Madrid, Comunidad de":"Madrid","Asturias, Principado de":"Asturias","Balears, Illes":"Illes Balears","Comunitat Valenciana":"Comunidade Valenciana","Murcia, Región de":"Murcia","Navarra, Comunidad Foral de":"Navarra","Rioja, La": "La Rioja","EE.UU.":"Estados Unidos"},inplace=True)
    print(df)



    
    
    # df_galicia=df[df.Zona=="Galicia"]
    df_españa=df[df.Zona=="España"]
    df_resto=df[df.Zona.isin(["Europa","Mundo"])]

    destino=gpd.GeoSeries([Point(-8.775,42.32)],crs="EPSG:4326")
    compute_distances = lambda x: x.distance(destino)[0]

 
    
    # #geomtries of spanish provinces for galician data
    # gdf=gpd.read_file("/home/usuario/OneDrive/curso_qgis/P05.09/provincias.shp")
    # gdf["centroid"]=gdf.geometry.centroid
    # islas_cies= islas_cies.to_crs(gdf.crs)
    # gdf["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf["centroid"])))
    # df_galicia=pd.merge(df_galicia,gdf[["provincia","cd_prov","nut2","centroid","geometry","Income","Población","distance (km)"]],left_on="Lugar",right_on="provincia",how="left")
    # print(df_galicia)

    #geometries of autonomous communities for spanish data
    gdf=gpd.read_file("/home/usuario/OneDrive/geo_data/Concellos/CCAA.shp")
    gdf=gdf.to_crs(destino.crs)
    gdf["centroid"]=gdf.geometry.centroid
    gdf["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf["centroid"])))
    df_españa=pd.merge(df_españa,gdf[["text","centroid","geometry","Income","Población","distance (km)"]],left_on="Lugar",right_on="text",how="left")
    print(df_españa)

    #geometry of countries for world data
    gdf=gpd.read_file("/home/usuario/OneDrive/geo_data/shp_mapa_paises_mundo_2014/Mapa_paises_mundo.shp")
    islas_cies=destino.to_crs("EPSG:3857")
    gdf=gdf.to_crs("EPSG:3857")
    gdf["centroid"]=gdf.geometry.centroid
    gdf["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf["centroid"])))
    gdf["Income"]=gdf["Income ($)"]*1/1.137
    gdf=gdf[["CNTR_ID","PAIS","distance (km)","Income","geometry"]]
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
    df=pd.concat([df_españa[["Año","Lugar","Numero","Income","Población","distance (km)"]],
                   df_resto[["Año","Lugar","Numero","Income","Población","distance (km)"]]])
    print(df[["Lugar","Numero"]])
    print(df)

    df.to_csv("tourist_origins_distances.csv",index=False)