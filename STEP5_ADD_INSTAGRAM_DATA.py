import pandas as pd
import sys
sys.path.append("../4kstogram")
from stogram import Stogram

if __name__== "__main__":

    #get Instagram user days for each park
    path="/home/usuario/Imágenes/4K Stogram/Islas Atlánticas de Galicia"
    con=Stogram(path,"Islas Atlánticas de Galicia")
    pud=con.photo_user_days(period="M")
    vud=con.video_user_days(period="M")
    df=pd.concat([pud,vud])
    df_islas_atlanticas=df.groupby(by=["date","SITE_NAME"],as_index=False).sum(numeric_only=True)
    

    path="/home/usuario/Imágenes/4K Stogram/Timanfaya"
    con=Stogram(path,"Timanfaya")
    pud=con.photo_user_days(period="M")
    vud=con.video_user_days(period="M")
    df=pd.concat([pud,vud])
    df_Timanfaya=df.groupby(by=["date","SITE_NAME"],as_index=False).sum(numeric_only=True)
    

    path="/home/usuario/Imágenes/4K Stogram/Tablas de Daimiel"
    con=Stogram(path,"Tablas de Daimiel")
    pud=con.photo_user_days(period="M")
    vud=con.video_user_days(period="M")
    df=pd.concat([pud,vud])
    df_daimiel=df.groupby(by=["date","SITE_NAME"],as_index=False).sum(numeric_only=True)

    path="/home/usuario/Imágenes/4K Stogram/Monfragüe"
    con=Stogram(path,"Monfragüe")
    pud=con.photo_user_days(period="M")
    vud=con.video_user_days(period="M")
    df=pd.concat([pud,vud])
    df_mongrague=df.groupby(by=["date","SITE_NAME"],as_index=False).sum(numeric_only=True)
    print(df_mongrague)

    path="/home/usuario/Imágenes/4K Stogram/Cabañeros"
    con=Stogram(path,"Cabañeros")
    pud=con.photo_user_days(period="M")
    vud=con.video_user_days(period="M")
    df=pd.concat([pud,vud])
    df_cabañeros=df.groupby(by=["date","SITE_NAME"],as_index=False).sum(numeric_only=True)
    print(df_cabañeros)

    path="/home/usuario/Imágenes/4K Stogram/Cabrera"
    con=Stogram(path,"Archipiélago de Cabrera")
    pud=con.photo_user_days(period="M")
    vud=con.video_user_days(period="M")
    df=pd.concat([pud,vud])
    df_cabrera=df.groupby(by=["date","SITE_NAME"],as_index=False).sum(numeric_only=True)
    print(df_cabrera)
    
    #concatenate all df's of each park in a single one
    df_instagram=pd.concat([df_islas_atlanticas,df_Timanfaya,df_daimiel,df_mongrague,df_cabañeros,df_cabrera])
    
    #some rows are missing, let us substitute this missin data with zeros
    date_idx=pd.date_range("01-01-2015","31-12-2022",freq="M").to_period("M") #we create a fully complete date range index
    arrays=[date_idx,df_instagram.SITE_NAME.unique()] #names of the parks
    index=pd.MultiIndex.from_product(arrays,names=["date","SITE_NAME"]) #we create a double index: parks and month
    df_instagram.set_index(["date","SITE_NAME"],inplace=True) #set date and park columns as index in the df
    df_instagram=df_instagram.reindex(index) #peform reindex
    df_instagram.reset_index(inplace=True) #recover the columns
    df_instagram.loc[df_instagram.IUD.isna(),"IUD"]=0 #set missing observations as zero
    df_instagram.to_csv("/home/usuario/Documentos/recreation/IUD.csv",index=False)


    #load reacreation data
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation_1concello.csv")
    df.Date=pd.to_datetime(df.Date)
    df.Date=df.Date.dt.to_period("M")
    print(df.info())
    
    #merge usind left keys (visitation data)
    df=df.merge(df_instagram,left_on=["SITE_NAME","Date"],right_on=["SITE_NAME","date"],how="left")
    df.drop(columns=["SITE_NAME","views","date"],inplace=True) #drop irrelevant columns
    print(df.info())
    df.to_csv("/home/usuario/Documentos/recreation/recreation_INE_FUD_IUD_1concello.csv",index=False)


