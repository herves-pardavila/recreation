import geopandas as gpd
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.append("/home/usuario/OneDrive/geo_data")
from meteogalicia import Meteogalicia

def add_environmental_data(pud,variable_code,start_date,finish_date):
    """Adds meteogalicia records to the photo-user-days databse. This functions works
    only when no temporal dimension is added.

    Args:
        pud (geopandas.GeoDataFrame): database with the aoi geometries and information for each cell
        variable_code (str): Short name of the variable. Check available names here https://www.meteogalicia.gal/observacion/rede/parametrosIndex.action
        start_date (str): Beginning of the time series in format dd/mm/yyyy
        finish_date (str): End of the time series in format dd/mm/yyyy
    """
    
    #get meteogalicia data
    con=Meteogalicia()
    datos=con.get_stations_data(variable_code,start_date,finish_date,frequency="monthly")
    datos=datos[datos.CodeValidation.isin([1,5])]
    datos=datos.groupby(by=["Stations"],as_index=False).mean(numeric_only=True)
 

    #load previously computed elsewhere voronoi polygons for this variable
    voronoi=gpd.read_file("/home/usuario/OneDrive/geo_data/meteogalicia_data/voronoi_"+variable_code+".shp")

    #merge data to voronoi
    datos=datos.merge(voronoi[["idStation","geometry"]],on="idStation",how="right")
    datos=gpd.GeoDataFrame(datos,crs=voronoi.crs,geometry=datos.geometry)

    #spatial overlay between environmental data and PUD data
    
    pud["index_visitation"]=pud.index
    datos.to_crs(pud.crs,inplace=True)
    overlay=gpd.overlay(datos,pud,how="intersection")
   #some gridded cell can overlay with more than  one voronoi polygon, we compute the area of overlay
    overlay["area"]=overlay["geometry"].area/10**6
    #we sort this area
    overlay.sort_values(by=["area"],inplace=True)
    #and we keep the largest overlap for each visitation gridded cell
    overlay.drop_duplicates(subset=["index_visitation"],keep="last",inplace=True)
    overlay.drop(["Stations","idStation","CodeValidation","altitude","idEstacion","lat","lon","index_visitation","geometry","area"],axis=1,inplace=True)
    overlay=pd.DataFrame(overlay)
    
    #recover geometries
    overlay=overlay.merge(pud[["FID","geometry"]],on="FID",how="left")
    overlay=gpd.GeoDataFrame(overlay,crs=pud.crs,geometry=overlay.geometry)

    #change  column names
    overlay.rename(columns={"Value":variable_code},inplace=True)

    return overlay





if __name__== "__main__":

    gdf_pud=gpd.read_file("PUD.shp")
    gdf=add_environmental_data(gdf_pud,"TA_AVG_1.5m","01/08/2019","31/12/2022")
    gdf=add_environmental_data(gdf,"TA_MAX_1.5m","01/08/2019","31/12/2022")
    gdf=add_environmental_data(gdf,"TA_MIN_1.5m","01/08/2019","31/12/2022")
    gdf=add_environmental_data(gdf,"PP_SUM_1.5m","01/08/2019","31/12/2022")
    gdf=add_environmental_data(gdf,"VV_AVG_2m","01/08/2019","31/12/2022")
    gdf=add_environmental_data(gdf,"HSOL_SUM_1.5m","01/08/2019","31/12/2022")
    gdf=add_environmental_data(gdf,"PRED_AVG_1.5m","01/08/2019","31/12/2022")
    print(gdf.columns)
    gdf.to_file("PUD.shp",index=False)

    

    

