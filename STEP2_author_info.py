import sys
sys.path.append("/home/usuario/OneDrive/recreation")
from flickr_photos import FlickrPhotos

if __name__== "__main__":

    con=FlickrPhotos("clipped_photos.csv")
    print(con.df)
    author_info=con.author_information(50)
    print(author_info)
    delete_authors=["SINDO MOSTEIRO","lebeauserge.es","Contando Estrelas","Bibliotecas Municipais da Coruña","Alberto Segade"]
    con.delete_users(delete_authors)
    print(con.df)
    con.df.to_csv("cleaned_photos.csv",index=False)
