import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    df=pd.read_excel("/home/usuario/Documentos/recreation/Islas Atlánticas/INE_visitantes_Cies.xlsx",decimal=",")
    df=df.set_index(" ").rename({' ':"Date"}).stack().reset_index(name="turistas").rename(columns={"level_1":"Provincia"})
    df.columns=["Date", "provincia", "turistas"]
    #print(df)
    gdf=gpd.read_file("/home/usuario/OneDrive/curso_qgis/P05.09/provincias.shp")
    gdf["centroid"]=gdf.geometry.centroid
    #print(gdf.info())
    #print(gdf.crs)

    df=pd.merge(df,gdf[["provincia","cd_prov","nut2","centroid","geometry","Income"]],on="provincia")
    
    #print(df.cd_prov.unique())

    islas_cies=gpd.GeoSeries([Point(-8.7,42.2)],crs="EPSG:4326")
    #print(islas_cies.crs)
    #islas_cies.set_crs("EPSG:4326",inplace=True)
    islas_cies=islas_cies.to_crs("epsg:25830")
   # print(islas_cies)

    compute_distances = lambda x: x.distance(islas_cies)[0]
    distances=map(compute_distances,df["centroid"])
    distances=list(distances)
    
    df["distance (km)"]=1e-3*np.array(distances)
    df=gpd.GeoDataFrame(df,crs="EPSG:25830",geometry=df.centroid)

    #print(df.provincia.unique())
    #print(df)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    df.plot(ax=ax, column="distance (km)")
    islas_cies.plot(ax=ax,color="red")
    df.plot(ax=ax, column="centroid",color="black")
    plt.show()


    #add population
    pop=pd.read_excel("/home/usuario/Documentos/recreation/poblacion_provincias_españolas.xlsx")
    newdf=pop.stack().reset_index(name="Provincia").rename(columns={"level_1":"Provincia","Provincia":"Poblacion"})
    newdf.drop(0,axis=0,inplace=True)

    codes= lambda x : x[0:2]
    newdf["cd_prov"]=list(map(codes,newdf.Provincia))
    newdf=newdf[["cd_prov","Poblacion"]]

    newdf=pd.merge(df[["Date","provincia","turistas","cd_prov","distance (km)","Income"]],newdf,on="cd_prov",how="left")
    newdf.rename(columns={"provincia":"origen"},inplace=True)
    newdf.Date=pd.to_datetime(newdf.Date,format="%YM%m").dt.to_period("M")
    #print(newdf)

    #turismo receptor
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2019.xlsx",sheet_name=None)
    df_receptor_2019=pd.concat([df_receptor["m07_2019"],df_receptor["m08_2019"],df_receptor["m09_2019"],df_receptor["m10_2019"],df_receptor["m11_2019"],df_receptor["m12_2019"]])
    df_receptor_2019=df_receptor_2019[df_receptor_2019.mun_dest=="Vigo"]
    df_receptor_2019=df_receptor_2019.groupby(["mes","mun_dest","mun_dest_cod","pais_orig"],as_index=False).sum(numeric_only=True)
    print(df_receptor_2019)
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2020.xlsx",sheet_name=None)
    df_receptor_2020=pd.concat([df_receptor["m01_2020"],df_receptor["m02_2020"],df_receptor["m03_2020"],df_receptor["m04_2020"],df_receptor["m05_2020"],df_receptor["m07_2020"],df_receptor["m07_2020"],df_receptor["m08_2020"],df_receptor["m09_2020"],df_receptor["m10_2020"],df_receptor["m11_2020"],df_receptor["m12_2020"]])
    df_receptor_2020=df_receptor_2020[df_receptor_2020.mun_dest=="Vigo"]
    df_receptor_2020=df_receptor_2020.groupby(["mes","mun_dest","mun_dest_cod","pais_orig"],as_index=False).sum(numeric_only=True)
    print(df_receptor_2020)
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2021.xlsx",sheet_name=None)
    df_receptor_2021=pd.concat([df_receptor["m01_2021"],df_receptor["m02_2021"],df_receptor["m03_2021"],df_receptor["m04_2021"],df_receptor["m05_2021"],df_receptor["m07_2021"],df_receptor["m07_2021"],df_receptor["m08_2021"],df_receptor["m09_2021"],df_receptor["m10_2021"],df_receptor["m11_2021"],df_receptor["m12_2021"]])
    df_receptor_2021=df_receptor_2021[df_receptor_2021.mun_dest=="Vigo"]
    df_receptor_2021=df_receptor_2021.groupby(["mes","mun_dest","mun_dest_cod","pais_orig"],as_index=False).sum(numeric_only=True)
    print(df_receptor_2021)
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2022.xlsx",sheet_name=None)
    df_receptor_2022=pd.concat([df_receptor["m01_2022"],df_receptor["m02_2022"],df_receptor["m03_2022"],df_receptor["m04_2022"],df_receptor["m05_2022"],df_receptor["m07_2022"],df_receptor["m07_2022"],df_receptor["m08_2022"],df_receptor["m09_2022"],df_receptor["m10_2022"],df_receptor["m11_2022"],df_receptor["m12_2022"]])
    df_receptor_2022=df_receptor_2022[df_receptor_2022.mun_dest=="Vigo"]
    df_receptor_2022=df_receptor_2022.groupby(["mes","mun_dest","mun_dest_cod","pais_orig"],as_index=False).sum(numeric_only=True)
    print(df_receptor_2022)
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2023.xlsx",sheet_name=None)
    df_receptor_2023=pd.concat([df_receptor["m01_2023"],df_receptor["m02_2023"],df_receptor["m03_2023"],df_receptor["m04_2023"],df_receptor["m05_2023"],df_receptor["m07_2023"],df_receptor["m07_2023"],df_receptor["m08_2023"],df_receptor["m09_2023"],df_receptor["m10_2023"],df_receptor["m11_2023"],df_receptor["m12_2023"]])
    df_receptor_2023=df_receptor_2023[df_receptor_2023.mun_dest=="Vigo"]
    df_receptor_2023=df_receptor_2023.groupby(["mes","mun_dest","mun_dest_cod","pais_orig"],as_index=False).sum(numeric_only=True)
    print(df_receptor_2023)
    
    dfreceptor=pd.concat([df_receptor_2019,df_receptor_2020,df_receptor_2021,df_receptor_2022,df_receptor_2023],ignore_index=True)
    print(dfreceptor)
    dfreceptor=dfreceptor[~dfreceptor.isin(['Total','Total Europa','Total Centroamérica y Caribe', 'Total Oceanía' , 'Total América', 'Total América del Norte', 'Total Asia', 'Total Sudamérica', 'Total Unión Europea', 'Total África'])]
    #df_receptor_2022.rename(columns={"mun_dest_cod":"dest_cod","turistas":"turistas_extranjeros"},inplace=True)
    print(dfreceptor.pais_orig.unique())

    gdf_world=gpd.read_file("/home/usuario/OneDrive/geo_data/shp_mapa_paises_mundo_2014/Mapa_paises_mundo.shp")
    print(gdf_world.info())
    print(gdf_world)
    print(gdf_world.PAIS.unique())
    print(gdf_world.crs)

    islas_cies=islas_cies.to_crs("EPSG:3857")
    gdf_world=gdf_world.to_crs("EPSG:3857")

    gdf_world["centroid"]=gdf_world.geometry.centroid
    gdf_world["distance (km)"]=1e-3*np.array(list(map(compute_distances,gdf_world["centroid"])))
    gdf_world["Income"]=gdf_world["Income ($)"]*1/1.137
    gdf_world=gdf_world[["CNTR_ID","PAIS","distance (km)","Income","geometry"]]
    

    print(gdf_world[["PAIS","distance (km)"]])

    centroides_mundo=gpd.GeoDataFrame(data=gdf_world,crs="EPSG:3857",geometry=gdf_world.centroid)
    centroides_mundo.to_file("/home/usuario/OneDrive/geo_data/centroides_mundo.shp")

    #merge
    dfreceptor.loc[dfreceptor.pais_orig=="Canada","pais_orig"]="Canadá"
    dfreceptor.loc[dfreceptor.pais_orig=="EE.UU.","pais_orig"]="Estados Unidos"
    dfreceptor.loc[dfreceptor.pais_orig=="Emiratos arabes unidos","pais_orig"]="Emiratos Árabes Unidos"
    dfreceptor.loc[dfreceptor.pais_orig=="Hungria","pais_orig"]="Hungría"
    dfreceptor.loc[dfreceptor.pais_orig=="Mexico","pais_orig"]="Méjico"
    dfreceptor.loc[dfreceptor.pais_orig=="Peru","pais_orig"]="Perú"
    dfreceptor.loc[dfreceptor.pais_orig=="Rumania","pais_orig"]="Rumanía"
    dfreceptor.loc[dfreceptor.pais_orig=="Bulgaria","pais_orig"]="Bulgaria"
    dfreceptor.loc[dfreceptor.pais_orig=="Sudafrica","pais_orig"]="Sudáfrica"
    dfreceptor.loc[dfreceptor.pais_orig=="Turquia","pais_orig"]="Turquía"

    dfreceptor=dfreceptor.merge(gdf_world[["CNTR_ID","PAIS","distance (km)","Income"]],left_on="pais_orig",right_on="PAIS",how="left")
    dfreceptor["Date"]=pd.to_datetime(dfreceptor.mes).dt.to_period("M")
    dfreceptor.rename(columns={"pais_orig":"origen"},inplace=True)
    dfreceptor=dfreceptor[["Date","origen","CNTR_ID","turistas","distance (km)","Income"]]

    print(dfreceptor)
    print(newdf)

    final_df=pd.concat([newdf[["Date","origen","turistas","distance (km)","Poblacion","Income"]],dfreceptor])
    print(final_df)
    final_df.to_csv("tourist_origins_distances.csv",index=False)


    # centroides_españa=df

    # fig2=plt.figure()
    # ax2=fig2.add_subplot(111)
    # gdf_world.plot(ax=ax2)
    # #centroides_mundo.plot(ax=ax2,column="distance (km)")
    # #centroides_españa.plot(ax=ax2,column="distance (km)")
    # gdf.plot(ax=ax2)
    # plt.show()