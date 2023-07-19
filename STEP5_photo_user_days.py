import sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("/home/usuario/OneDrive/recreation")
from flickr_photos import FlickrPhotos
from STEP2_build_database import reindex_dataframe, add_environmental_data
sys.path.append("/home/usuario/OneDrive/geo_data")
from geo_funciones import create_square_grid

if __name__== "__main__":

    #grid the area of interest
    create_square_grid("/home/usuario/OneDrive/recreation/qgis/dissolvedaoi.shp",5000,"griddedaoi.shp")

    #compute photo-user
    con=FlickrPhotos("cleaned_photos.csv")
    print(con.df)
    pu=con.photo_user()
    
    #compute photo-user-days
    pu.date=pd.to_datetime(pu.date)
    pu["year"]=pu.date.dt.year
    pu["month"]=pu.date.dt.month
    pu["day_of_year"]=pu.date.dt.dayofyear
    aoi=gpd.read_file("griddedaoi.shp")
    print("Numero de celdas en el area de interés=",len(aoi))
    geopud=con.photo_user_days(pu,aoi)
    print(len(geopud.FID.unique()))
    #we add missing dates
    newgeopud=reindex_dataframe(geopud[["FID","date","PUD"]],"FID","date",[],"2019-08-01","2022-12-31")
    newgeopud.loc[newgeopud.PUD.isna(),"PUD"]=0

    #group by months ans sum
    newgeopud.date=pd.to_datetime(newgeopud.date)
    newgeopud["year"]=newgeopud.date.dt.year
    newgeopud["month"]=newgeopud.date.dt.month
    gdf_pud=newgeopud.groupby(by=["year","month","FID"],as_index=False).sum(numeric_only=True)
    
    #group by cells and compute average PUD
    gdfpud=gdf_pud.groupby(by="FID",as_index=False).mean(numeric_only=True)
    gdfpud=gdfpud.merge(aoi[["FID","geometry"]],on="FID",how="left")
    gdfpud=gpd.GeoDataFrame(gdfpud,crs=aoi.crs,geometry=gdfpud.geometry)
    print(gdfpud)
    gdfpud=gdfpud[gdfpud.PUD > 0]
    gdfpud.plot("PUD",legend=True)
    plt.show()
    gdfpud[["FID","PUD","geometry"]].to_file("PUD.shp",index=False)


    # #seasonality
    # pud_summer=gdf_pud[gdf_pud.month.isin([6,7,8,9])]
    # pud_rest=gdf_pud[~gdf_pud.month.isin([6,7,8,9])]
    # pud_summer=pud_summer.groupby(by="FID",as_index=False).mean(numeric_only=True)
    # pud_rest=pud_rest.groupby(by="FID",as_index=False).mean(numeric_only=True)
    # pud_summer=pud_summer.merge(aoi[["FID","geometry"]],on="FID",how="left")
    # pud_rest=pud_rest.merge(aoi[["FID","geometry"]],on="FID",how="left")
    # pud_summer=gpd.GeoDataFrame(pud_summer,crs=aoi.crs,geometry=pud_summer.geometry)
    # pud_rest=gpd.GeoDataFrame(pud_rest,crs=aoi.crs,geometry=pud_rest.geometry)

    # pud_summer=pud_summer[(pud_summer.PUD !=0) & (pud_summer.PUD < 14)]
    # pud_rest=pud_rest[(pud_rest.PUD !=0) & (pud_rest.PUD < 14)]
    # pud_summer.plot("PUD",legend=True)
    # pud_rest.plot("PUD",legend=True)
    # plt.show()





    