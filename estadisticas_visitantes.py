import pandas as pd
import geopandas as gpd

if __name__ == "__main__":

    concellos=gpd.read_file("/home/usuario/OneDrive/geo_data/Concellos/concellos_costeiros.shp")
    concellos.CODIGOINE=concellos.CODIGOINE.astype(int)
    codigos_concellos=concellos.CODIGOINE.unique()
    print(codigos_concellos)


    #datos ine 2022
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2022.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m01_2022"],df_receptor["m02_2022"],df_receptor["m03_2022"],df_receptor["m04_2022"],df_receptor["m05_2022"],df_receptor["m06_2022"],df_receptor["m07_2022"],df_receptor["m08_2022"],df_receptor["m09_2022"],df_receptor["m10_2022"],df_receptor["m11_2022"],df_receptor["m12_2022"]])
   
    df_receptor.mun_dest_cod=df_receptor.mun_dest_cod.astype(int)
    df_recpetor=df_receptor[df_receptor.pais_orig!="Total"]
    df_receptor=df_receptor[df_receptor.mun_dest_cod.isin(codigos_concellos)]
    df_receptor=df_receptor[df_receptor.pais_orig.isin(["Total América del Norte","Total Asia","Total Europa",
                                                        "Total Sudamérica","Total Unión Europea","Total África",
                                                        "Total Centroamérica y Caribe","Total Oceanía"])]
    df_receptor=df_receptor.groupby(by=["mes","pais_orig","mun_dest"],as_index=False).mean(numeric_only=True)


    df_interno=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_interno_mun_2022.xlsx",sheet_name=None)
    df_interno=pd.concat([df_interno["2022-01"],df_interno["2022-02"],df_interno["2022-03"],df_interno["2022-04"],df_interno["2022-05"],df_interno["2022-06"],df_interno["2022-07"],df_interno["2022-08"],df_interno["2022-09"],df_interno["2022-10"],df_interno["2022-11"],df_interno["2022-12"]])
    df_interno.dest_cod=df_interno.dest_cod.astype(int)
    
    df_interno=df_interno[df_interno.dest_cod.isin(codigos_concellos)]
    df_interno=df_interno.groupby(by=["mes","prov_orig","dest"],as_index=False).sum(numeric_only=True)
    df_gallegos=df_interno[df_interno.prov_orig.isin(["Coruña, A","Lugo","Ourense","Pontevedra"])]
    df_españoles=df_interno[~df_interno.prov_orig.isin(["Coruña, A","Lugo","Ourense","Pontevedra"])]
    df_españoles=df_españoles.groupby(by=["mes","dest"],as_index=False).sum(numeric_only=True)
    df_españoles["origen"]="España"
    


    df_receptor.rename(columns={"pais_orig":"origen","mun_dest":"destino"},inplace=True)
    df_gallegos.rename(columns={"prov_orig":"origen","dest":"destino"},inplace=True)
    df_gallegos.origen=df_gallegos.origen.astype(str)
    df_españoles.rename(columns={"dest":"destino"},inplace=True)

    df_interno=pd.concat([df_gallegos,df_españoles])
    print(df_interno)
    df_ine_2022=pd.concat([df_interno[["mes","origen","destino","turistas"]],df_receptor[["mes","origen","destino","turistas"]]])
    print(df_ine_2022)

    #datos ine 2019
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2019.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m07_2019"],df_receptor["m08_2019"],df_receptor["m09_2019"],df_receptor["m10_2019"],df_receptor["m11_2019"],df_receptor["m12_2019"]])
   
    df_receptor.mun_dest_cod=df_receptor.mun_dest_cod.astype(int)
    df_recpetor=df_receptor[df_receptor.pais_orig!="Total"]
    df_receptor=df_receptor[df_receptor.mun_dest_cod.isin(codigos_concellos)]
    df_receptor=df_receptor[df_receptor.pais_orig.isin(["Total América del Norte","Total Asia","Total Europa",
                                                        "Total Sudamérica","Total Unión Europea","Total África",
                                                        "Total Centroamérica y Caribe","Total Oceanía"])]
    df_receptor=df_receptor.groupby(by=["mes","pais_orig","mun_dest"],as_index=False).mean(numeric_only=True)


    df_interno=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_interno_mun_2019.xlsx",sheet_name=None)
    df_interno=pd.concat([df_interno["2019-07"],df_interno["2019-08"],df_interno["2019-09"],df_interno["2019-10"],df_interno["2019-11"],df_interno["2019-12"]])
    df_interno.dest_cod=df_interno.dest_cod.astype(int)
    
    df_interno=df_interno[df_interno.dest_cod.isin(codigos_concellos)]
    df_interno=df_interno.groupby(by=["mes","prov_orig","dest"],as_index=False).sum(numeric_only=True)
    df_gallegos=df_interno[df_interno.prov_orig.isin(["Coruña, A","Lugo","Ourense","Pontevedra"])]
    df_españoles=df_interno[~df_interno.prov_orig.isin(["Coruña, A","Lugo","Ourense","Pontevedra"])]
    df_españoles=df_españoles.groupby(by=["mes","dest"],as_index=False).sum(numeric_only=True)
    df_españoles["origen"]="España"
    


    df_receptor.rename(columns={"pais_orig":"origen","mun_dest":"destino"},inplace=True)
    df_gallegos.rename(columns={"prov_orig":"origen","dest":"destino"},inplace=True)
    df_gallegos.origen=df_gallegos.origen.astype(str)
    df_españoles.rename(columns={"dest":"destino"},inplace=True)

    df_interno=pd.concat([df_gallegos,df_españoles])
    print(df_interno)
    df_ine_2019=pd.concat([df_interno[["mes","origen","destino","turistas"]],df_receptor[["mes","origen","destino","turistas"]]])
    print(df_ine_2019)


    #datos ine 2023
    df_receptor=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_receptor_mun_2023.xlsx",sheet_name=None)
    df_receptor=pd.concat([df_receptor["m01_2023"],df_receptor["m02_2023"],df_receptor["m03_2023"],df_receptor["m04_2023"],df_receptor["m05_2023"],df_receptor["m06_2023"],df_receptor["m07_2023"],df_receptor["m08_2023"],df_receptor["m09_2023"],df_receptor["m10_2023"],df_receptor["m11_2023"],df_receptor["m12_2023"]])
   
    df_receptor.mun_dest_cod=df_receptor.mun_dest_cod.astype(int)
    df_recpetor=df_receptor[df_receptor.pais_orig!="Total"]
    df_receptor=df_receptor[df_receptor.mun_dest_cod.isin(codigos_concellos)]
    df_receptor=df_receptor[df_receptor.pais_orig.isin(["Total América del Norte","Total Asia","Total Europa",
                                                        "Total Sudamérica","Total Unión Europea","Total África",
                                                        "Total Centroamérica y Caribe","Total Oceanía"])]
    df_receptor=df_receptor.groupby(by=["mes","pais_orig","mun_dest"],as_index=False).mean(numeric_only=True)


    df_interno=pd.read_excel("/home/usuario/Documentos/recreation/exp_tmov_interno_mun_2023.xlsx",sheet_name=None)
    df_interno=pd.concat([df_interno["2023-01"],df_interno["2023-02"],df_interno["2023-03"],df_interno["2023-04"],df_interno["2023-05"],df_interno["2023-06"],df_interno["2023-07"],df_interno["2023-08"],df_interno["2023-09"],df_interno["2023-10"],df_interno["2023-11"],df_interno["2023-12"]])
    df_interno.dest_cod=df_interno.dest_cod.astype(int)
    
    df_interno=df_interno[df_interno.dest_cod.isin(codigos_concellos)]
    df_interno=df_interno.groupby(by=["mes","prov_orig","dest"],as_index=False).sum(numeric_only=True)
    df_gallegos=df_interno[df_interno.prov_orig.isin(["Coruña, A","Lugo","Ourense","Pontevedra"])]
    df_españoles=df_interno[~df_interno.prov_orig.isin(["Coruña, A","Lugo","Ourense","Pontevedra"])]
    df_españoles=df_españoles.groupby(by=["mes","dest"],as_index=False).sum(numeric_only=True)
    df_españoles["origen"]="España"
    


    df_receptor.rename(columns={"pais_orig":"origen","mun_dest":"destino"},inplace=True)
    df_gallegos.rename(columns={"prov_orig":"origen","dest":"destino"},inplace=True)
    df_gallegos.origen=df_gallegos.origen.astype(str)
    df_españoles.rename(columns={"dest":"destino"},inplace=True)

    df_interno=pd.concat([df_gallegos,df_españoles])
    print(df_interno)
    df_ine_2023=pd.concat([df_interno[["mes","origen","destino","turistas"]],df_receptor[["mes","origen","destino","turistas"]]])
    print(df_ine_2023)

    df=pd.concat([df_ine_2019,df_ine_2022,df_ine_2023])
    print(df)
    df.to_csv("/home/usuario/Documentos/recreation/PCCMMG/origen_turistas.csv",index=False)
 
 
 