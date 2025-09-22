# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 16:42:31 2025

@author: dherves
"""

import pandas as pd
import os

if __name__== "__main__":
   
    path=r"F:/doctorado/"
    
    path_income_tables=path+"/recreation/ZonalTravelCost/datos_municipales/RBMPP/"
    
    df=pd.DataFrame()
    
    for root,dird,files in os.walk(path_income_tables):
        for file in files:
            print(file)
            newdf = pd.read_csv(root+file,encoding="latin-1",sep=";",decimal=".",
                                names=["Municipios","Distritos","Secciones",
                                       "Indicadores de renta media y mediana",
                                       "Periodo","Total"],skiprows=1,na_values=[".",".."])
            newdf=newdf.loc[newdf["Indicadores de renta media y mediana"]=="Renta bruta media por persona"]
            newdf=newdf.loc[newdf.Periodo==2021]
            newdf2=newdf.groupby(by=["Municipios","Indicadores de renta media y mediana","Periodo"],as_index=False).mean(numeric_only=True)
            df=pd.concat([df,newdf2],ignore_index=True)
        df=df[["Municipios","Indicadores de renta media y mediana","Periodo",
               "Total"]]
    
    obtener_codigos = lambda x :int(x[0:6])
    
    df["mun_orig_cod"]=list(map(obtener_codigos,df.Municipios))
      
    #Hacemos el merge con los datos del STEP1

    df2 = pd.read_csv(path +"recreation/ZonalTravelCost/datos_municipales/3travel_cost_Ons.csv")    
    
    df=pd.merge(df2,df,how="left",on="mun_orig_cod")
    
    df.rename(columns = {"Total":"RBMPP"},inplace=True)
    
    df.to_csv(df[["Lugar","Año","Zona","turistasINE","POBLACION_","distance (km)",
                 "RBMPP"]].to_csv(path  +"recreation/ZonalTravelCost/datos_municipales/3travel_cost_Ons_with_Income.csv"),
              index=False)
    
