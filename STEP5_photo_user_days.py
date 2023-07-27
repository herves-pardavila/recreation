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
    create_square_grid("/home/usuario/OneDrive/recreation/qgis/dissolvedaoi.shp",10000,"griddedaoi.shp")

    #compute photo-user
    con=FlickrPhotos("./cleaned_photos.csv")
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
    newgeopud.date=pd.to_datetime(newgeopud.date)
    pandemia=pd.date_range("2020-03-10","2021-09-01")
    newgeopud=newgeopud[~newgeopud.date.isin(pandemia)]
    newgeopud.loc[newgeopud.PUD.isna(),"PUD"]=0

    #we group by months
    newgeopud["Month"]=newgeopud.date.dt.month
    newgeopud["Year"]=newgeopud.date.dt.year
    newgeopud=newgeopud.groupby(by=["FID","Month","Year"],as_index=False).sum(numeric_only=True)

    gdfpud=newgeopud.merge(aoi[["FID","geometry"]],on="FID",how="left")
    gdfpud=gpd.GeoDataFrame(gdfpud,crs=aoi.crs,geometry=gdfpud.geometry)
    gdfpud["date"]=pd.to_datetime(dict(year=gdfpud.Year,month=gdfpud.Month,day=1))
    print(gdfpud)
    gdfpud["date"]=gdfpud.date.astype(str)
    df=pd.DataFrame(gdfpud[["FID","PUD","date"]])
    print(df)
    df.to_csv("PUD.csv",index=False)








    