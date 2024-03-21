import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

if __name__== "__main__":

    #load geometries and general info of municipalities
    df=pd.read_csv("/home/usuario/OneDrive/geo_data/Concellos/MUNICIPIOS.csv") #important data of each municipality: population, mean altitude, surface...
    gdf_peninsula=gpd.read_file("/home/usuario/OneDrive/geo_data/Concellos/SHP_ETRS89/recintos_municipales_inspire_peninbal_etrs89/recintos_municipales_inspire_peninbal_etrs89.shp")
    
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

   
    #merge geometries and general info
    gdf_peninsula=gdf_peninsula[["NAMEUNIT","new_codes","geometry"]].merge(df[["COD_PROV","PROVINCIA","NOMBRE_ACTUAL","POBLACION_MUNI","SUPERFICIE","PERIMETRO","ALTITUD","new_codes"]],on="new_codes",how="left")
    gdf_peninsula["POBLACION_MUNI"]=gdf_peninsula["POBLACION_MUNI"].astype(float)
    print(gdf_peninsula.info())
    

    #datos ine 2022
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2022.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m01_2022"],df_receptor["m02_2022"],df_receptor["m03_2022"],df_receptor["m04_2022"],df_receptor["m05_2022"],df_receptor["m06_2022"],df_receptor["m07_2022"],df_receptor["m08_2022"],df_receptor["m09_2022"],df_receptor["m10_2022"],df_receptor["m11_2022"],df_receptor["m12_2022"]])
    df_recpetor=df_receptor[df_receptor.pais_orig=="Total"]
    df_receptor=df_receptor.groupby(["mes","mun_dest_cod"],as_index=False).sum(numeric_only=True)
    df_receptor.rename(columns={"mun_dest_cod":"dest_cod","turistas":"turistas_extranjeros"},inplace=True)

    df_interno=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_interno_mun_2022.xlsx",sheet_name=None)
    df_interno=pd.concat([df_interno["2022-01"],df_interno["2022-02"],df_interno["2022-03"],df_interno["2022-04"],df_interno["2022-05"],df_interno["2022-06"],df_interno["2022-07"],df_interno["2022-08"],df_interno["2022-09"],df_interno["2022-10"],df_interno["2022-11"],df_interno["2022-12"]])
    df_interno=df_interno.groupby(["dest_cod","mes"],as_index=False).sum(numeric_only=True)

    df_ine_2022=pd.merge(df_interno[["dest_cod","mes","turistas"]],df_receptor[["dest_cod","mes","turistas_extranjeros"]],on=["dest_cod","mes"],how="outer")
    df_ine_2022["turistas_total"]=df_ine_2022.turistas+df_ine_2022.turistas_extranjeros
    print(df_ine_2022.info())

    #datos ine 2021
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2021.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m01_2021"],df_receptor["m02_2021"],df_receptor["m03_2021"],df_receptor["m04_2021"],df_receptor["m05_2021"],df_receptor["m06_2021"],df_receptor["m07_2021"],df_receptor["m08_2021"],df_receptor["m09_2021"],df_receptor["m10_2021"],df_receptor["m11_2021"],df_receptor["m12_2021"]])
    df_recpetor=df_receptor[df_receptor.pais_orig=="Total"]
    df_receptor=df_receptor.groupby(["mes","mun_dest_cod"],as_index=False).sum(numeric_only=True)
    df_receptor.rename(columns={"mun_dest_cod":"dest_cod","turistas":"turistas_extranjeros"},inplace=True)

    df_interno=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_interno_mun_2021.xlsx",sheet_name=None)
    df_interno=pd.concat([df_interno["2021-01"],df_interno["2021-02"],df_interno["2021-03"],df_interno["2021-04"],df_interno["2021-05"],df_interno["2021-06"],df_interno["2021-07"],df_interno["2021-08"],df_interno["2021-09"],df_interno["2021-10"],df_interno["2021-11"],df_interno["2021-12"]])
    df_interno=df_interno.groupby(["dest_cod","mes"],as_index=False).sum(numeric_only=True)

    df_ine_2021=pd.merge(df_interno[["dest_cod","mes","turistas"]],df_receptor[["dest_cod","mes","turistas_extranjeros"]],on=["dest_cod","mes"],how="outer")
    df_ine_2021["turistas_total"]=df_ine_2021.turistas+df_ine_2021.turistas_extranjeros
    print(df_ine_2021.info())

    #datos ine 2020
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2020.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m01_2020"],df_receptor["m02_2020"],df_receptor["m03_2020"],df_receptor["m04_2020"],df_receptor["m05_2020"],df_receptor["m06_2020"],df_receptor["m07_2020"],df_receptor["m08_2020"],df_receptor["m09_2020"],df_receptor["m10_2020"],df_receptor["m11_2020"],df_receptor["m12_2020"]])
    df_recpetor=df_receptor[df_receptor.pais_orig=="Total"]
    df_receptor=df_receptor.groupby(["mes","mun_dest_cod"],as_index=False).sum(numeric_only=True)
    df_receptor.rename(columns={"mun_dest_cod":"dest_cod","turistas":"turistas_extranjeros"},inplace=True)

    df_interno=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_interno_mun_2020.xlsx",sheet_name=None)
    df_interno=pd.concat([df_interno["2020-01"],df_interno["2020-02"],df_interno["2020-03"],df_interno["2020-04"],df_interno["2020-05"],df_interno["2020-06"],df_interno["2020-07"],df_interno["2020-08"],df_interno["2020-09"],df_interno["2020-10"],df_interno["2020-11"],df_interno["2020-12"]])
    df_interno=df_interno.groupby(["dest_cod","mes"],as_index=False).sum(numeric_only=True)

    df_ine_2020=pd.merge(df_interno[["dest_cod","mes","turistas"]],df_receptor[["dest_cod","mes","turistas_extranjeros"]],on=["dest_cod","mes"],how="outer")
    df_ine_2020["turistas_total"]=df_ine_2020.turistas+df_ine_2020.turistas_extranjeros
    print(df_ine_2020.info())

    #datos ine 2019
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2019.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m07_2019"],df_receptor["m08_2019"],df_receptor["m09_2019"],df_receptor["m10_2019"],df_receptor["m11_2019"],df_receptor["m12_2019"]])
    df_recpetor=df_receptor[df_receptor.pais_orig=="Total"]
    df_receptor=df_receptor.groupby(["mes","mun_dest_cod"],as_index=False).sum(numeric_only=True)
    df_receptor.rename(columns={"mun_dest_cod":"dest_cod","turistas":"turistas_extranjeros"},inplace=True)

    df_interno=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_interno_mun_2019.xlsx",sheet_name=None)
    df_interno=pd.concat([df_interno["2019-07"],df_interno["2019-08"],df_interno["2019-09"],df_interno["2019-10"],df_interno["2019-11"],df_interno["2019-12"]])
    df_interno=df_interno.groupby(["dest_cod","mes"],as_index=False).sum(numeric_only=True)

    df_ine_2019=pd.merge(df_interno[["dest_cod","mes","turistas"]],df_receptor[["dest_cod","mes","turistas_extranjeros"]],on=["dest_cod","mes"],how="outer")
    df_ine_2019["turistas_total"]=df_ine_2019.turistas+df_ine_2019.turistas_extranjeros
    print(df_ine_2019.info())


    df_ine=pd.concat([df_ine_2022,df_ine_2021,df_ine_2020,df_ine_2019])




    df_peninsula=pd.merge(df_ine,gdf_peninsula,left_on="dest_cod",right_on="new_codes",how="right")
    df_peninsula.dropna(subset="NOMBRE_ACTUAL",inplace=True)
    print("\n")
    print("==================================================================================")
    print(df_peninsula)
    print(df_peninsula.info())
    print("\n")
    print("==================================================================================")
 
    #save raw data
    df_peninsula.drop(columns="geometry",inplace=True)
    df=df_peninsula
    print(df)
    df.to_csv("/home/usuario/Documentos/recreation/PCCMMG/turismo.csv",index=False)