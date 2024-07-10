import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

if __name__== "__main__":

    #load geometries and general info of municipalities
    df=pd.read_csv("/home/usuario/OneDrive/geo_data/Concellos/MUNICIPIOS.csv") #important data of each municipality: population, mean altitude, surface...
    gdf_peninsula=gpd.read_file("/home/usuario/OneDrive/geo_data/Concellos/SHP_ETRS89/recintos_municipales_inspire_peninbal_etrs89/recintos_municipales_inspire_peninbal_etrs89.shp")
    gdf_canarias=gpd.read_file("/home/usuario/OneDrive/geo_data/Concellos/SHP_REGCAN95/recintos_municipales_inspire_canarias_regcan95/recintos_municipales_inspire_canarias_regcan95.shp")

    #fix codes for general info
    new_codes= lambda x: str(x)[:-6]
    df["new_codes"]=list(map(new_codes,df.COD_INE))
    df.new_codes=df.new_codes.astype(int)
    print(df.sort_values(by="new_codes")[["NOMBRE_ACTUAL","new_codes"]])
    #remove strange data
    df.loc[2630,"POBLACION_MUNI"]=np.nan
    df.loc[2675,"POBLACION_MUNI"]=np.nan

    #fix codes for the geometries
    ine_codes= lambda x : str(x)[6:]
    gdf_peninsula["new_codes"]=list(map(ine_codes,gdf_peninsula.NATCODE))
    gdf_peninsula.new_codes=gdf_peninsula.new_codes.astype(int)
    gdf_canarias["new_codes"]=list(map(ine_codes,gdf_canarias.NATCODE))
    gdf_canarias.new_codes=gdf_canarias.new_codes.astype(int)
   
    #merge geometries and general info
    gdf_peninsula=gdf_peninsula[["NAMEUNIT","new_codes","geometry"]].merge(df[["COD_PROV","PROVINCIA","NOMBRE_ACTUAL","POBLACION_MUNI","SUPERFICIE","PERIMETRO","ALTITUD","new_codes"]],on="new_codes",how="left")
    gdf_canarias=gdf_canarias[["NAMEUNIT","new_codes","geometry"]].merge(df[["COD_PROV","PROVINCIA","NOMBRE_ACTUAL","POBLACION_MUNI","SUPERFICIE","PERIMETRO","ALTITUD","new_codes"]],on="new_codes",how="left")
    
    
    gdf_peninsula["POBLACION_MUNI"]=gdf_peninsula["POBLACION_MUNI"].astype(float)
    gdf_canarias["POBLACION_MUNI"]=gdf_canarias["POBLACION_MUNI"].astype(float)
    print(gdf_peninsula.info())
    print(gdf_canarias.info())

    # turismo receptor

    #datos ine 2023
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2023.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m01_2023"],df_receptor["m02_2023"],df_receptor["m03_2023"],df_receptor["m04_2023"],df_receptor["m05_2023"],df_receptor["m06_2023"],df_receptor["m07_2023"],df_receptor["m08_2023"],df_receptor["m09_2023"],df_receptor["m10_2023"],df_receptor["m11_2023"],df_receptor["m12_2023"]])
    #df_receptor=df_receptor.groupby(["mes","mun_dest_cod"],as_index=False).sum(numeric_only=True)
    df_receptor.rename(columns={"mun_dest_cod":"dest_cod","turistas":"turistas_extranjeros","pais_orig":"Origen","mun_dest":"Destino"},inplace=True)
    df_ine_2023=df_receptor
    print(df_ine_2023)
    print(df_ine_2023.info())
    
    
    
    #datos ine 2022
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2022.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m01_2022"],df_receptor["m02_2022"],df_receptor["m03_2022"],df_receptor["m04_2022"],df_receptor["m05_2022"],df_receptor["m06_2022"],df_receptor["m07_2022"],df_receptor["m08_2022"],df_receptor["m09_2022"],df_receptor["m10_2022"],df_receptor["m11_2022"],df_receptor["m12_2022"]])
    #df_receptor=df_receptor.groupby(["mes","mun_dest_cod"],as_index=False).sum(numeric_only=True)
    df_receptor.rename(columns={"mun_dest_cod":"dest_cod","turistas":"turistas_extranjeros","pais_orig":"Origen","mun_dest":"Destino"},inplace=True)
    df_ine_2022=df_receptor
    print(df_ine_2022)
    print(df_ine_2022.info())

    #datos ine 2021
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2021.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m01_2021"],df_receptor["m02_2021"],df_receptor["m03_2021"],df_receptor["m04_2021"],df_receptor["m05_2021"],df_receptor["m06_2021"],df_receptor["m07_2021"],df_receptor["m08_2021"],df_receptor["m09_2021"],df_receptor["m10_2021"],df_receptor["m11_2021"],df_receptor["m12_2021"]])
    #df_receptor=df_receptor.groupby(["mes","mun_dest_cod"],as_index=False).sum(numeric_only=True)
    df_receptor.rename(columns={"mun_dest_cod":"dest_cod","turistas":"turistas_extranjeros","pais_orig":"Origen","mun_dest":"Destino"},inplace=True)
    df_ine_2021=df_receptor

    #datos ine 2020
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2020.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m01_2020"],df_receptor["m02_2020"],df_receptor["m03_2020"],df_receptor["m04_2020"],df_receptor["m05_2020"],df_receptor["m06_2020"],df_receptor["m07_2020"],df_receptor["m08_2020"],df_receptor["m09_2020"],df_receptor["m10_2020"],df_receptor["m11_2020"],df_receptor["m12_2020"]])
    #df_receptor=df_receptor.groupby(["mes","mun_dest_cod"],as_index=False).sum(numeric_only=True)
    df_receptor.rename(columns={"mun_dest_cod":"dest_cod","turistas":"turistas_extranjeros","pais_orig":"Origen","mun_dest":"Destino"},inplace=True)
    df_ine_2020=df_receptor


    # #datos ine 2019
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2019.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m07_2019"],df_receptor["m08_2019"],df_receptor["m09_2019"],df_receptor["m10_2019"],df_receptor["m11_2019"],df_receptor["m12_2019"]])
    #df_receptor=df_receptor.groupby(["mes","mun_dest_cod"],as_index=False).sum(numeric_only=True)
    df_receptor.rename(columns={"mun_dest_cod":"dest_cod","turistas":"turistas_extranjeros","pais_orig":"Origen","mun_dest":"Destino"},inplace=True)
    df_ine_2019=df_receptor
    print(df_ine_2019.info())

    #concatenate all outer tourism
    df_ine=pd.concat([df_ine_2023,df_ine_2022,df_ine_2021,df_ine_2020,df_ine_2019])
    df_ine.mes=pd.to_datetime(df_ine.mes).dt.to_period("M")
    
    print(df_ine)
    print(df_ine.info())

    #Load inner tourism
    df_interno=pd.read_csv("../../Documentos/turismo_interno2019-2023.csv",thousands=".")
    print(df_interno.info())
    df_interno["mes"]=pd.to_datetime(df_interno.Date,format="%YM%m").dt.to_period("M")
    codes=lambda x : x[0:6]
    df_interno["dest_cod"]=list(map(codes,df_interno.Destino))
    df_interno.turistas=df_interno.turistas.astype(int)
    #df_interno=df_interno[["mes","dest_cod","turistas"]].groupby(by=["dest_cod","mes"],as_index=False).sum(numeric_only=True)
    df_interno=df_interno[df_interno.dest_cod!='Total ']
    df_interno.dest_cod=df_interno.dest_cod.astype(int)
    print(df_interno.info())
    print(df_ine.info())
    print(df_interno.Destino)
    print(df_ine.Destino)

    #merge inner and outer tourism
    df_ine=pd.merge(df_interno[["Origen","dest_cod","mes","turistas"]],df_ine[["Origen","dest_cod","mes","turistas_extranjeros"]],on=["dest_cod","mes","Origen"],how="outer")
    print(df_ine)

    df_peninsula=pd.merge(df_ine,gdf_peninsula,left_on="dest_cod",right_on="new_codes",how="right")
    df_peninsula.dropna(subset="NOMBRE_ACTUAL",inplace=True)
    df_canarias=pd.merge(df_ine,gdf_canarias,left_on="dest_cod",right_on="new_codes",how="right")
    df_canarias.dropna(subset="NOMBRE_ACTUAL",inplace=True)
    print("\n")
    print("==================================================================================")
    print(df_peninsula)
    print(df_peninsula.info())
    print("\n")
    print("==================================================================================")
    print(df_canarias)
    print(df_canarias.info())

    #save raw data
    df_peninsula.drop(columns="geometry",inplace=True)
    df_canarias.drop(columns="geometry",inplace=True)
    df=pd.concat([df_peninsula,df_canarias])
    print(df)
    df.to_csv("/home/usuario/Documentos/recreation/turismo_with_origins.csv",index=False)