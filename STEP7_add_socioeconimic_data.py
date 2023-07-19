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

    gdf_pud=gpd.read_file("PUD.shp")

    #add leng of trial routes and roads
    grid=length_of_lines("/home/usuario/OneDrive/geo_data/caminos_naturales/all_trial_network.shp",gdf_pud,spatial_indicator="FID",name="trial_len",output_file="/home/usuario/OneDrive/recreation/qgis/recreation_sdata.shp",save=False)
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
    print(grid)

    
    
    #For the following variables. It is easier to load the intermediate shapfiles that InVEST produces, with the alrady desired metric and perform spatial overlays so we keep adding more variables to the database
    predictor_data=gpd.read_file("/home/usuario/OneDrive/recreation/InVEST/scenario_data/predictor_data_all_variables.shp")  
    print(predictor_data.columns)

    
  
    #add protected zones
    grid=spatial_overlay(grid,predictor_data[["protected","geometry"]])
    #add pointcount hotels
    grid=spatial_overlay(grid,predictor_data[["hotels","geometry"]])
    #add elevation
    grid=spatial_overlay(grid,predictor_data[["dem","geometry"]])
    #beach area 
    grid=spatial_overlay(grid,predictor_data[["beach","geometry"]])
    #number of hotels, hostels, campings...ç
    grid=spatial_overlay(grid,predictor_data[["hotels","geometry"]])
    #nautic port
    grid=spatial_overlay(grid,predictor_data[["nautic","geometry"]])
    #lulc
    grid=spatial_overlay(grid,predictor_data[["arable","built","crops","forest","grassland","intertidal","marshes","rocks","scrub","geometry"]])
    print(grid)

    grid.fillna(value=0,inplace=True)
    grid.to_file("PUD.shp",index=False)
    grid.to_csv("PUD.csv",index=False)





    
 