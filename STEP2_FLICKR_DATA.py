import geopandas as gpd
import pandas as pd
import sys
sys.path.append("/home/usuario/OneDrive/recreation")
from flickr_photos import FlickrPhotos


if __name__ == "__main__":

    con=FlickrPhotos("/home/usuario/github/recreationdb/photos_parques_naturales.csv")
    #clip by mask of the area of interest
    con.clip("/home/usuario/OneDrive/geo_data/protected_zones/parques_nacionales.shp")
    clipped_df=con.gdf
    con.print_general_info()
    con.df=clipped_df

    #author info
    df_autores=con.author_information(100)
    print(df_autores[["Name","Photos"]])
    df_autores[["Name","Photos"]].to_csv("autores.csv",index=False)
    
    #tag info
    tags=con.get_tags(con.df)
    df_tags=con.word_count(tags)
    df_tags_100=df_tags[df_tags.Count>100]
    print(df_tags_100)
    con.word_cloud(tags)

    #title info
    title_words=con.get_title_words(con.df)
    df_title_words=con.word_count(title_words)
    titles100=df_title_words[df_title_words.Count>100]
    #titles100.to_csv("title_words_more_than_100_times.csv",index=True)
    print(titles100)
    con.word_cloud(title_words)

    pu=con.photo_user()
    aoi=gpd.read_file("/home/usuario/OneDrive/geo_data/protected_zones/parques_nacionales.shp")
    pud=con.photo_user_days(pu,aoi,"SITE_NAME")
    pud=pud[["SITE_NAME","Date","views","PUD"]]
    print(pud)
    #the other data we are going to use, INE tourist and park visitors are monthly, therefore, we group by months
    pud.Date=pd.to_datetime(pud.Date)
    pud.Date=pud.Date.dt.to_period("M")
    pud=pud.groupby(by=["SITE_NAME","Date"],as_index=False).sum(numeric_only=True)
    print(pud)
    pud.to_csv("/home/usuario/Documentos/recreation/pud.csv",index=False)