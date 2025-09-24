#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 20:00:51 2025

@author: david
"""
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    path=r"F:/doctorado/"
    #load the data
    #df=pd.read_csv(path+"3travel_cost_Ons.csv")
    df=pd.read_csv(path+"recreation/ZonalTravelCost/3travel_cost_Aiguestortes_ready.csv")
    
    df=df[df.Año.isin([2022,2023])]
    df=df.groupby(["Lugar","Zona"],as_index=False).mean(numeric_only=True)
    
    comunidad_autonoma="Cataluña"
    
    Q_comunidad=df[df.Lugar == comunidad_autonoma][["Numero","turistasINE"]]
    Q_españa=df.groupby(by="Zona",as_index=True).sum(numeric_only=True).loc["España",["Numero","turistasINE"]]
    Q_resto=df.groupby(by="Zona",as_index=True).sum(numeric_only=True).loc[["Mundo","Europa"],["Numero","turistasINE"]].sum()
    
    print("Datos on-site")
    var="Numero"
    print("Porcentaje de visitantes de la misma comunidad=",float(Q_comunidad[var])/(float(Q_españa[var])+float(Q_resto[var])))
    print("Porcentaje de visitantes de otra comunidad=",float(Q_españa[var]-Q_comunidad[var])/float(Q_españa[var]+Q_resto[var]))
    print("Porcentaje de visitantes no españoles=",float(Q_resto[var])/float(Q_españa[var]+Q_resto[var]))
    
    
    print("Datos moviles")
    var="turistasINE"
    print("Porcentaje de visitantes de la misma comunidad=",float(Q_comunidad[var])/(float(Q_españa[var])+float(Q_resto[var])))
    print("Porcentaje de visitantes de otra comunidad=",float(Q_españa[var]-Q_comunidad[var])/float(Q_españa[var]+Q_resto[var]))
    print("Porcentaje de visitantes no españoles=",float(Q_resto[var])/float(Q_españa[var]+Q_resto[var]))
