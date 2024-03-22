import geopandas as gpd
import sys
import pandas as pd
sys.path.append("/home/usuario/OneDrive/recreation")
from flickr_photos import FlickrPhotos


if __name__ == "__main__":

    con=FlickrPhotos("/home/usuario/github/recreationdb/photos_GZ_estatico.csv")
    #clip by mask of the area of interest
    con.clip("/home/usuario/OneDrive/geo_data/Concellos/Concellos_IGN.shp")
    clipped_df=con.gdf
    con.print_general_info()
    con.df=clipped_df

    #author info
    df_autores=con.author_information(100)
    print(df_autores[["Name","Photos"]])
    df_autores[["Name","Photos"]].to_csv("autores.csv",index=False)
    df=con.delete_users(["Alberto Segade","Bibliotecas Municipais da Coruña","SINDO MOSTEIRO","https://adventuresportsmedia.com","Festival Atlantica","PLANIFICACIÓN DEPOURENSE"])
    
    # #tag info
    # tags=con.get_tags(con.df)
    # df_tags=con.word_count(tags)
    # df_tags_100=df_tags[df_tags.Count>100]
    # print(df_tags_100)
    # con.word_cloud(tags)

    # #title info
    # title_words=con.get_title_words(con.df)
    # df_title_words=con.word_count(title_words)
    # #titles100=df_title_words[df_title_words.Count>100]
    # #titles100.to_csv("title_words_more_than_100_times.csv",index=True)
    # titles50=df_title_words[df_title_words.Count>50]
    # print(titles50)
    # con.word_cloud(title_words)
    # print(df.info())

    pu=con.photo_user()
    aoi=gpd.read_file("/home/usuario/OneDrive/geo_data/Concellos/Concellos_IGN.shp")
    pud=con.photo_user_days(pu,aoi,"CODIGOINE")
    pud=pud[["CODIGOINE","Date","views","PUD"]]
    print(pud.info())
    pud.to_csv("/home/usuario/Documentos/recreation/PCCMMG/pud.csv",index=False)