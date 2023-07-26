import geopandas as gpd
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.append("/home/usuario/OneDrive/geo_data")
from geo_funciones import create_square_grid
from geo_funciones import length_of_lines
from geo_funciones import distance_to_points, point_count


def spatial_overlay(layer,secondary_layer):
    if layer.crs!= secondary_layer.crs:
        secondary_layer.to_crs(layer.crs,inplace=True)
    
    overlay=gpd.overlay(secondary_layer,layer,how="intersection")
    overlay["area (km2)"]=overlay["geometry"].area/1e6
    overlay.sort_values(by=["area (km2)"],inplace=True)
    overlay.drop_duplicates(subset=["FID"],keep="last",inplace=True)
    overlay=pd.merge(overlay,layer,on="FID",how="right",suffixes=["_x",None])
    overlay.drop(overlay.filter(regex='_x$').columns, axis=1, inplace=True)
    overlay.drop("area (km2)",axis=1,inplace=True)
    return overlay

if __name__== "__main__":


    aoi=gpd.read_file("griddedaoi.shp")
    print(aoi)

    #add leng of trial routes and bicycle routes
    grid=length_of_lines("/home/usuario/OneDrive/geo_data/caminos_naturales/all_trial_network.shp",aoi,spatial_indicator="FID",name="trial_len",output_file="/home/usuario/OneDrive/recreation/qgis/recreation_sdata.shp",save=False)
    grid=length_of_lines("/home/usuario/OneDrive/geo_data/caminos_naturales/bicycle_routesOSM.shp",grid,spatial_indicator="FID",name="bike_len",output_file="/home/usuario/OneDrive/recreation/qgis/recreation_sdata.shp",save=False)
    print(grid)
    #add distance to point layers
    grid=distance_to_points("/home/usuario/OneDrive/geo_data/OpenStreetMaps/tnodes.shp",grid,"Com-nodes (m)",output_file="/home/usuario/OneDrive/recreation/qgis/recreation_sdata.shp",save=False)
    grid=distance_to_points("/home/usuario/OneDrive/geo_data/transport/railway_network.shp",grid,"dist-train",output_file="/home/usuario/OneDrive/recreation/qgis/recreation_sdata.shp",save=False)
    
    grid=distance_to_points("/home/usuario/OneDrive/geo_data/OpenStreetMaps/lighthouse_galicia.shp",grid,"lighthouse (m)",output_file="/home/usuario/OneDrive/recreation/qgis/recreation_sdata.shp",save=False)
    grid=distance_to_points("/home/usuario/OneDrive/geo_data/OpenStreetMaps/cities_galicia.shp",grid,"city (m)",output_file="/home/usuario/OneDrive/recreation/qgis/recreation_sdata.shp",save=False)
    grid=distance_to_points("/home/usuario/OneDrive/geo_data/OpenStreetMaps/towns_galicia.shp",grid,"town (m)",output_file="/home/usuario/OneDrive/recreation/qgis/recreation_sdata.shp",save=False)
    print(grid)

  
    
    #add number of restuarants in each cell
    restaurants=gpd.read_file("/home/usuario/OneDrive/geo_data/OpenStreetMaps/restaurants_galicia.shp")
    grid=point_count(grid,restaurants,"FID","Value")
    grid.loc[grid.restaurant.isna(),"restaurant"]=0
    viewpoints=gpd.read_file("/home/usuario/OneDrive/geo_data/OpenStreetMaps/viewpoints_galicia.shp")
    grid=point_count(grid,viewpoints,"FID","Value")
    hotels=gpd.read_file("/home/usuario/OneDrive/recreation/qgis/OSMtourism.shp")
    grid=point_count(grid,hotels,"FID","name")


    
  
    #add protected zones
    pzones=gpd.read_file("/home/usuario/OneDrive/recreation/InVEST/protected_zones_only.shp")
    overlay=gpd.overlay(aoi,pzones,how="intersection")
    overlay["area (m2)"]=overlay.geometry.area
    grid=grid.merge(overlay [["FID","area (m2)"]],on="FID",how="left")
    grid.rename(columns={"area (m2)":"Protected area (m2)"},inplace=True)

    #add beach area
    beach=gpd.read_file("/home/usuario/OneDrive/geo_data/2021_03_26_GEAMA_PIMA_Universidades/vector_data/playas_GEAMA.shp")
    overlay=gpd.overlay(aoi,beach,how="intersection")
    overlay["area (m2)"]=overlay.geometry.area
    grid=grid.merge(overlay [["FID","area (m2)"]],on="FID",how="left")
    grid.rename(columns={"area (m2)":"Beach area (m2)"},inplace=True)

    #nautic port
    port=gpd.read_file("/home/usuario/OneDrive/geo_data/OpenStreetMaps/leisure/nautic_ports_OSM.shp")
    overlay=gpd.overlay(aoi,port,how="intersection")
    overlay["area (m2)"]=overlay.geometry.area
    grid=grid.merge(overlay [["FID","area (m2)"]],on="FID",how="left")
    grid.rename(columns={"area (m2)":"Port area (m2)"},inplace=True)


    del(port)
    del(beach)
    del(pzones)
    del(hotels)
    del(restaurants)
    del(viewpoints)
    # #lulc
    # lulcs=["arable_land","built_land","crops","dunes","forest","grassland","intertidal_flats","marshes","rocks"]
    # #lulcs=["marshes","intertidal_flats"]
    # for lulc in lulcs:
    #     print(lulc)
    #     gdf=gpd.read_file("/home/usuario/OneDrive/recreation/InVEST/lulc/"+lulc+".shp")
    #     overlay=gpd.overlay(aoi,gdf,how="intersection")
    #     del(gdf)
    #     overlay["area (m2)"]=overlay.geometry.area
    #     grid=grid.merge(overlay [["FID","area (m2)"]],on="FID",how="left")
    #     grid.rename(columns={"area (m2)":lulc+"area (m2)"},inplace=True)
        


    grid["Area (km2)"]=grid.geometry.area/1e6
    grid.fillna(value=0,inplace=True)
    grid.to_file("PUD.shp")
    print(grid)
    pud=pd.read_csv("PUD.csv")
    pud=pud.merge(grid,on="FID",how="left")
    pud.drop(["geometry"],axis=1,inplace=True)
    print(pud)
    pud.to_csv("PUD.csv",index=False)





    
 