import sys
sys.path.append("/home/usuario/OneDrive/recreation")
from flickr_photos import FlickrPhotos

if __name__== "__main__":

    con=FlickrPhotos("cleaned_photos.csv")
    print(con.df)
    title_words=con.get_title_words(con.df)
    title_info=con.word_count(title_words)
    print(title_info[title_info.Count > 50])
    title_info.to_csv("title_info.csv")
    delete_titles=["airbus","monbus","boeing"]
    con.delete_titles(delete_titles)
    print(con.df)
    title_words=con.get_title_words(con.df)
    con.word_cloud(title_words)
    con.df.to_csv("cleaned_photos.csv",index=False)