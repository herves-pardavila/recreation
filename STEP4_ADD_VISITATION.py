import pandas as pd
import numpy as np

if __name__=="__main__":
    main_path="/media/david/EXTERNAL_USB/doctorado/"
    #load OAPNA data
    visitantes=pd.read_csv(main_path+"/recreation/visitantes_parques_naturales_2015_2023_limpio.csv")
    visitantes.Date=pd.to_datetime(visitantes.Date).dt.to_period("M")
    print(visitantes)
    #load tourist data
    df=pd.read_csv(main_path+"/recreation/recreation_INE_flickr.csv")
    print(df)
    df.Date=pd.to_datetime(df.Date).dt.to_period("M")
    
    #merge both datasets
    df=df.merge(visitantes,left_on=["SITE_NAME","Date"],right_on=["IdOAPN","Date"],how="right")
    print(df.info())
    print("=========================================================================")
    df.to_csv(main_path+"recreation/recreation.csv",index=False)
    print(df)



  
   
 

    

    

    