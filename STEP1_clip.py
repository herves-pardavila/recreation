import sys
sys.path.append("/home/usuario/OneDrive/recreation")
from flickr_photos import FlickrPhotos

if __name__== "__main__":

    con=FlickrPhotos("/home/usuario/OneDrive/recreation/INE/ine_prueba.csv")
    con.clip("/home/usuario/OneDrive/recreation/qgis/dissolved_aoi_only_land.shp")
    con.plot()
    df=con.gdf
    df.drop("geometry",axis=1,inplace=True)
    print(df)
    df.to_csv("clipped_photos.csv",index=False)