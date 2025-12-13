import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt
import pypopulation
if __name__ == "__main__":
    
    path=r"D:/doctorado/"
   

    df_interno=pd.read_csv(path+"recreation/ZonalTravelCost/datos_provinciales/turismo_interno_Bueu_Sanxenxo_origen_provincias_2022_2024.csv",
                           sep=";",encoding="latin-1",na_values=["","."]) #origenes por CCAA y paises
    
    df_interno=df_interno.loc[df_interno["Municipio de destino"]=="36004 Bueu"]
    df_interno = df_interno.groupby(by=["CCAA y provincia de origen.2","Periodo"],as_index=False).sum(numeric_only=True)
    df_interno["mes"] = df_interno["Periodo"].str.replace(r"M", "-", regex=True)
    df_interno.mes=pd.to_datetime(df_interno.mes,format="%Y-%m").dt.to_period("M")
    df_interno["Año"]=df_interno.mes.dt.year
    df_interno=df_interno[df_interno.Año.isin([2022,2023])] # ELEGIR CORRECTAMENTE LOS AÑOS
    df_interno["Zona"]= "España"    
    
    
    
    df_interno.rename(columns={"CCAA y provincia de origen.2":"Lugar","Total":"turistasINE"},inplace=True)  
    
    df_interno=df_interno[["mes","Año","Lugar","Zona","turistasINE"]]
    df_interno=df_interno.groupby(by=["Lugar","Año","Zona"],as_index=False).sum(numeric_only=True) #convertimos en datos anuales
    
    

    #PARTE GEO
    destino=gpd.GeoSeries([Point(-8.775,42.32)],crs="EPSG:4326") #destino Ons    
    destino= destino.to_crs("EPSG:3857")
    compute_distances = lambda x: x.distance(destino)[0]   
    #geometries of spanish provinces for galician data
    gdf=gpd.read_file(path+"recreation/ZonalTravelCost/datos_provinciales/Ons_geodata_provincias.gpkg")
    gdf.to_crs("EPSG:3857",inplace=True)
    gdf["centroid"]=gdf.geometry.centroid
    gdf=gdf[gdf.centroid!= None]
    gdf["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf["centroid"])))
   
    
   #UNIR LA PARTE GEO CON LA PARTE DE DATOS DE TURSMIO
   
    newdf=pd.merge(df_interno,gdf[["Provincias","CODIGOINE","Población",
                                  "RBMPP","distance (km)"]],
                                  left_on="Lugar",right_on="Provincias",
                                  how="left")
    newdf[["Lugar","Año","Zona","CODIGOINE","turistasINE","Población",
           "RBMPP","distance (km)"]].to_csv(path +"recreation/ZonalTravelCost/datos_provinciales/3travel_cost_Ons.csv",
                                                        index=False)
    
    newgdf=pd.merge(df_interno,gdf[["Provincias","CODIGOINE","Población",
                                  "RBMPP","distance (km)","geometry"]],
                                  left_on="Lugar",right_on="Provincias",
                                  how="right")
    
    

    newgdf= gpd.GeoDataFrame(data=newgdf[["Lugar","Año","Zona","CODIGOINE","turistasINE","Población",
            "RBMPP","distance (km)","geometry"]],crs=gdf.crs,geometry=newgdf.geometry)
    
    
    variable="RBMPP"
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    newgdf.plot(column=variable, ax= ax)
    plt.show()

    newgdf.to_file(path +"recreation/ZonalTravelCost/datos_provinciales/3travel_cost_Ons.gpkg",
                   driver="GPKG",index=False)
    
