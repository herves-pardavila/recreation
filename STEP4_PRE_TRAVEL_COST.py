import pandas as pd
import geopandas as gpd
# import numpy as np
# from patsy import dmatrices
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import statsmodels.formula.api as smf
# import statsmodels.graphics as smg
# from sklearn.metrics import r2_score
# from scipy import stats
# from datetime import datetime

if __name__== "__main__":
    path=r"F:/doctorado/"
    #load the data
    df=pd.read_csv(path+"recreation/ZonalTravelCost/3travel_cost_Ons.csv")
    #df=df[df.Año==2019]
    print(df)
    
    df["distance"]=df["distance (km)"].astype(float)
    df.loc[df.Lugar=="Andorra","RBMPP"]=44720/1.184
    df.RBMPP=df.RBMPP/365
    df["CT_(€)"]=0
    
    #cost of travel
    df["CT_(€)"]=df.distance*0.12
    df.loc[df.Zona=="España","CT_(€)"]=df.distance*0.26
    df.loc[df.Lugar.isin(["Andorra","Francia","Portugal"]),"CT_(€)"]=df.distance*0.26
    df.loc[df.Lugar.isin(["Reino Unido","Italia","Suiza","Bélgica","Irlanda","Países Bajos"]),"CT_(€)"]=df.distance*0.2
    

    #opoprtunity cost
    df["OC_(€)"]=2*(1/3)*df.RBMPP
    df.loc[df.Zona=="España","OC_(€)"]=(1/3)*df.RBMPP
    df.loc[df.Lugar.isin(["Francia","Andorra","Portugal"]),"OC_(€)"]=(1/3)*df.RBMPP
    
    df.turistasINE
   
    print(df)
    print(df.info())
    df["TC"]=df["CT_(€)"]+df["OC_(€)"]
    df.to_csv(path+"recreation/ZonalTravelCost/3travel_cost_Ons_ready.csv",index=False)

    # fig=plt.figure()
    # fig.suptitle("Demand Curve")
    # ax=fig.add_subplot(111)
    # ax.set_ylabel("Trip Cost (p)")
    # ax.set_xlabel("Trips (Q)")
    # ax.plot(df.Numero,df["TC"],"o",color="black",label="Real Data")
    # ax.plot(df.Numero,df["TC"],"o",color="black",label="Poisson or NB")
    # ax.plot(df.turistasINE,df["CT_(€)"]+df["OC_(€)"],"o",color="red",label="INE Data")
    # ax.plot(df.yhat_full,df["CT_(€)"]+df["OC_(€)"],"o",color="blue",label="Predictor")
    # ax.set_xlim(0,10e4)
    # fig.legend() 
    # plt.show()
    
    #============= PARTE GEO ==================================
    gdf_comunidades=gpd.read_file(path+"OneDrive/geo_data/Concellos/CCAA.shp")
    gdf_paises = gpd.read_file(path+"OneDrive/geo_data/shp_mapa_paises_mundo_2014/Mapa_paises_mundo.shp")
    
    newdf= pd.merge(df,gdf_paises[["PAIS","geometry"]],how="left",left_on = "Lugar",right_on = "PAIS")
    newdf = pd.merge(newdf, gdf_comunidades[["text","geometry"]],left_on=["Lugar"],right_on=["text"],how="outer")
    newdf.loc[newdf.geometry_x==None,"geometry_x"]=newdf.loc[newdf.geometry_x==None,"geometry_y"]
    newdf.drop(columns="geometry_y",inplace=True)
    
    new_gdf=gpd.GeoDataFrame(data=newdf,crs=gdf_comunidades.crs,geometry="geometry_x")
    new_gdf.to_file(path+"recreation/ZonalTravelCost/3travel_cost_Ons.gpkg",driver="GPKG")
