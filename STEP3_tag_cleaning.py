import sys
sys.path.append("/home/usuario/OneDrive/recreation")
from flickr_photos import FlickrPhotos

if __name__== "__main__":

    con=FlickrPhotos("cleaned_photos.csv")
    print(con.df)
    tags=con.get_tags(con.df)
    tag_info=con.word_count(tags)
    print(tag_info[tag_info.Count>50])
    tag_info[tag_info.Count>50].to_csv("tag_info.csv",index=True)
    delete_tags=["privado","pintura","bodegasdelpalaciodefefiñanes","adegasdepazodefefiñáns","2022asturiascantabria","autorretrato","bng","oviedo"]
    con.delete_tags(delete_tags)
    print(con.df)
    tags=con.get_tags(con.df)
    tag_info=con.word_count(tags)
    con.word_cloud(tags)
    con.df.to_csv("cleaned_photos.csv",index=False)
    