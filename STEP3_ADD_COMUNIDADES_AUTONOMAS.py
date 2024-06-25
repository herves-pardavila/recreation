import pandas as pd


if __name__== "__main__":

    df=pd.read_csv("/home/usuario/Documentos/recreation/monfragüe/INE_visitantes.csv")
    print(df.info())

    #add autonomous communities
    df["CCAA"]=" "
    #df.loc[df.prov_orig.isin(["Coruña, A"]),"CCAA"]="A coruña"
    #df.loc[df.prov_orig.isin(["Ourense"]),"CCAA"]="Ourense"
    #df.loc[df.prov_orig.isin(["Lugo"]),"CCAA"]="Lugo"
    df.loc[df.prov_orig.isin(["Coruña, A","Pontevedra","Lugo","Ourense"]),"CCAA"]="Galicia"
    df.loc[df.prov_orig.isin(["Asturias"]),"CCAA"]="Asturias"
    df.loc[df.prov_orig.isin(["Burgos","León","Palencia","Salamanca","Segovia","Zamora","Ávila","Soria","Valladolid"]),"CCAA"]="Castilla y León"
    df.loc[df.prov_orig.isin(["Cantabria"]),"CCAA"]="Cantabria"
    df.loc[df.prov_orig.isin(["Araba/Álava","Bizkaia","Gipuzkoa"]),"CCAA"]="País Vasco"
    df.loc[df.prov_orig.isin(["Navarra"]),"CCAA"]="Navarra"
    df.loc[df.prov_orig.isin(["Rioja, La"]),"CCAA"]="La Rioja"
    df.loc[df.prov_orig.isin(["Huesca","Zaragoza","Teruel"]),"CCAA"]="Aragón"
    df.loc[df.prov_orig.isin(["Lleida","Tarragona","Barcelona","Girona"]),"CCAA"]="Cataluña"
    df.loc[df.prov_orig.isin(["Alicante/Alacant","Castellón/Castelló","Valencia/València"]),"CCAA"]="Comunidade Valenciana"
    df.loc[df.prov_orig.isin(["Madrid"]),"CCAA"]="Madrid"
    df.loc[df.prov_orig.isin(["Albacete","Guadalajara","Cuenca","Ciudad Real","Toledo"]),"CCAA"]="Castilla - La Mancha"
    df.loc[df.prov_orig.isin(["Badajoz","Cáceres"]),"CCAA"]="Extremadura"
    df.loc[df.prov_orig.isin(["Almería","Cádiz","Córdoba","Granada","Huelva","Málaga","Sevilla","Jaén"]),"CCAA"]="Andalucía"
    df.loc[df.prov_orig.isin(["Murcia"]),"CCAA"]="Murcia"
    df.loc[df.prov_orig.isin(["Balears, Illes"]),"CCAA"]="Baleares"
    df.loc[df.prov_orig.isin(["Santa Cruz de Tenerife","Palmas, Las"]),"CCAA"]="Canarias"
    df.loc[df.prov_orig.isin(["Ceuta"]),"CCAA"]="Ceuta"
    df.loc[df.prov_orig.isin(["Melilla"]),"CCAA"]="Melilla"
    
    df.loc[pd.isna(df.prov_orig),"País"]="No España"
    df.loc[~pd.isna(df.prov_orig),"País"]="España"
    df.loc[pd.isna(df.prov_orig),"CCAA"]=df[pd.isna(df.prov_orig)]["pais_orig"]
    df.loc[pd.isna(df.prov_orig),"turistas"]=df[pd.isna(df.prov_orig)]["turistas_extranjeros"]
  

    
    df.Date=pd.to_datetime(df.Date)
    df["Year"]=df.Date.dt.year
    # location="Vigo"
    año=2020
    #df=df[df.NAMEUNIT=="Vigo"]
    df=df[df.Year==año]
    df=df[["Year","NAMEUNIT","SITE_NAME","CCAA","prov_orig","País","turistas"]]
    print(df)
    # #df_interior=
    #df=df.groupby(by=["CCAA","prov_orig","País","NAMEUNIT"],as_index=False).sum(numeric_only=True)
    #print(df)
    df=df[~df.CCAA.isin(['Total','Total Europa','Total Centroamérica y Caribe', 'Total Oceanía' , 'Total América', 'Total América del Norte', 'Total Asia', 'Total Sudamérica', 'Total Unión Europea', 'Total África'])]
    df=df.groupby(by=["CCAA","País"],as_index=False).sum(numeric_only=True)
    #df=df[df.País=="No España"]
    df=df.sort_values("turistas",ascending=False)
    print(df)
    print(df.CCAA.unique())
    print(df[df.País.isin(["España"])][["turistas"]].sum())
    

