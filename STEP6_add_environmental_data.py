import geopandas as gpd
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.append("/home/usuario/OneDrive/geo_data")
from meteogalicia import Meteogalicia

def add_environmental_data(pud,aoi,variable_code,start_date,finish_date):
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
    datos["date"]=pd.to_datetime(datos.Dates)
    datos.date=datos.date.astype(str)
    print(datos)
 

    #load previously computed elsewhere voronoi polygons for this variable
    voronoi=gpd.read_file("/home/usuario/OneDrive/geo_data/meteogalicia_data/voronoi_"+variable_code+".shp")

    # #merge data to voronoi
    # datos=datos.merge(voronoi[["idStation","geometry"]],on="idStation",how="left")
    # datos=gpd.GeoDataFrame(datos,crs=voronoi.crs,geometry=datos.geometry)
    # print(len(datos))

    #spatial overlay between environmental data and PUD data
    aoi["index_visitation"]=aoi.index
    voronoi.to_crs(aoi.crs,inplace=True)
    overlay=gpd.overlay(voronoi,aoi,how="intersection")
   #some gridded cell can overlay with more than  one voronoi polygon, we compute the area of overlay
    overlay["area"]=overlay["geometry"].area/10**6
    #we sort this area
    overlay.sort_values(by=["area"],inplace=True)
    #and we keep the largest overlap for each visitation gridded cell
    overlay.drop_duplicates(subset=["index_visitation"],keep="last",inplace=True)



    #add geometries and stations to PUD dataframe
    pud=pud.merge(overlay[["idStation","FID","geometry"]],on="FID",how="left")
    pud=pud.merge(datos[["idStation","date","Value"]],on=["idStation","date"],how="left")
    pud.drop(["idStation","geometry"],axis=1,inplace=True)
    pud.rename(columns={"Value":variable_code},inplace=True)


    return pud





if __name__== "__main__":

    pud=pd.read_csv("PUD.csv")
    aoi=gpd.read_file("griddedaoi.shp")
    pud=add_environmental_data(pud,aoi,"TA_AVG_1.5m","01/08/2019","31/12/2022")
    pud=add_environmental_data(pud,aoi,"TA_MAX_1.5m","01/08/2019","31/12/2022")
    pud=add_environmental_data(pud,aoi,"TA_MIN_1.5m","01/08/2019","31/12/2022")
    pud=add_environmental_data(pud,aoi,"PP_SUM_1.5m","01/08/2019","31/12/2022")
    pud=add_environmental_data(pud,aoi,"VV_AVG_2m","01/08/2019","31/12/2022")
    pud=add_environmental_data(pud,aoi,"HSOL_SUM_1.5m","01/08/2019","31/12/2022")
    pud=add_environmental_data(pud,aoi,"PR_AVG_1.5m","01/08/2019","31/12/2022")
    print(pud)
    #print(gdf.columns)
    pud.to_csv("PUD.csv",index=False)
    

    

    

