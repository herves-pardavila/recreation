#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:31:24 2025

@author: david
"""
import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
from sklearn.metrics import r2_score
from scipy import stats
from datetime import datetime


if __name__== "__main__":
    
    main_path="/media/david/EXTERNAL_USB/doctorado/"
    #prepare the data
    df=pd.read_csv(main_path+"recreation/recreation_ready.csv")
    df=df[df.Visitantes != 0]
    #df=df[df.IdOAPN.isin(["Timanfaya","Islas Atlánticas de Galicia","Tablas de Daimiel","Monfragüe","Cabañeros","Archipiélago de Cabrera"])]
    #df=df[df.Year.isin([2015,2016,2017,2018])]
    df.Date=df.Date.astype("category")
    df.Month=df.Month.astype("category")
    df.Year=df.Year.astype("category")
    df.IdOAPN=df.IdOAPN.astype("category")
    df.Season=df.Season.astype("category")
    df.covid=df.covid.astype("category")
    
    
    # print(df)
    # print(df.info())
    # sum_statistics=df[["Visitantes","PUD","turistas_total"]].describe()
    # sum_statistics=sum_statistics.round({"Visitantes":0,"PUD":1,"IUD":1,"turistas_total":0,"turistas_corregido":0})
    # print(sum_statistics)
    # sum_statistics.to_csv(main_path+"recreation/imagenes_paper/summary_statistics.csv",index=True)
    
    # print(df[["Visitantes","PUD","IUD","turistas_total","turistas_corregido"]].corr("spearman"))
    
    dfmean=df[["IdOAPN","Visitantes","PUD","IUD","turistas_total","turistas_corregido","Month"]].groupby(by=["IdOAPN","Month"],as_index=False).mean()
    dfvar=df[["IdOAPN","Visitantes","Month"]].groupby(by=["IdOAPN","Month"],as_index=False).var()
    dfmean=dfmean.merge(dfvar,on=["IdOAPN","Month"],how="inner")
    dfmean["mu/sigma"]=dfmean.Visitantes_x/dfmean.Visitantes_y
    print(dfmean)
    print(dfmean.describe())
    dfmean=dfmean.groupby(by="IdOAPN",as_index=False).mean(numeric_only=True)
    dfmean.rename(columns={"Visitantes_x":"Visitors (mu)","PUD":"FUD","turistas_total":"MPUD","turistas_corregido":"correctedMPUD"},inplace=True)
    dfmean=dfmean.round({"Visitors (mu)":0,"FUD":1,"IUD":1,"MPUD":0,"correctedMPUD":0,"mu/sigma":4})
    dfmean.sort_values(by="Visitors (mu)",inplace=True,ascending=False)
    print(dfmean)
    dfmean.to_csv(main_path+"recreation/imagenes_paper/tabla_overdispersion.csv",index=False)
    
    
    print(df.Month.unique())
    dfanual=df.groupby(by=["IdOAPN"],as_index=False).sum(numeric_only=True)
    #print(dfanual.Month.unique())
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.loglog(dfanual.Visitantes,dfanual.PUD,"*",label="Flickr PUD")
    ax.loglog(dfanual.Visitantes,dfanual.turistas_total,"*",label="INE data")
    ax.loglog(dfanual.Visitantes,dfanual.turistas_corregido,"*",label="INE data corregido")
    ax.loglog(dfanual.Visitantes,dfanual.IUD,"*",label="Instagram")
    ax.loglog(np.arange(1e5,8e7,10),np.arange(1e5,8e7,10),label="1-1 line")
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.PUD,dfanual.IdOAPN)]
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.turistas_total,dfanual.IdOAPN)]
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.turistas_corregido,dfanual.IdOAPN)]
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.IUD,dfanual.IdOAPN)]
    ax.set_xlabel("Visitors")
    ax.set_ylabel("Estimated User-generated-data visitors")
    fig.legend()
    plt.show()
 
    #Monthly visitation per park and seasonality
    dfmensual=df.groupby(by=["IdOAPN","Month"],as_index=False).sum(numeric_only=True)
    print(dfmensual.Month.unique())
    print(dfmensual)
    dfmensualA=dfmensual[dfmensual.IdOAPN=="Aigüestortes i Estany de Sant Maurici"]
    dfmensualB=dfmensual[dfmensual.IdOAPN=="Archipiélago de Cabrera"]
    dfmensualC=dfmensual[dfmensual.IdOAPN=="Cabañeros"]
    dfmensualD=dfmensual[dfmensual.IdOAPN=="Caldera de Taburiente"]
    dfmensualE=dfmensual[dfmensual.IdOAPN=="Doñana"]
    dfmensualF=dfmensual[dfmensual.IdOAPN=="Garajonay"]
    dfmensualG=dfmensual[dfmensual.IdOAPN=="Islas Atlánticas de Galicia"]
    dfmensualH=dfmensual[dfmensual.IdOAPN=="Monfragüe"]
    dfmensualI=dfmensual[dfmensual.IdOAPN=="Ordesa y Monte Perdido"]
    dfmensualJ=dfmensual[dfmensual.IdOAPN=="Picos de Europa"]
    dfmensualK=dfmensual[dfmensual.IdOAPN=="Sierra Nevada"]
    print(dfmensualK)
    dfmensualL=dfmensual[dfmensual.IdOAPN=="Sierra de Guadarrama"]
    dfmensualM=dfmensual[dfmensual.IdOAPN=="Sierra de las Nieves"]
    dfmensualN=dfmensual[dfmensual.IdOAPN=="Tablas de Daimiel"]
    dfmensualO=dfmensual[dfmensual.IdOAPN=="Teide National Park"]
    dfmensualP=dfmensual[dfmensual.IdOAPN=="Timanfaya"]
 
    fig2=plt.figure()
    ax2=fig2.subplot_mosaic("""ABCD
                         EFGH
                         IJKL
                         MNOP""")
 
    fig2.subplots_adjust(hspace=0.45,wspace=0.3,top=0.9,bottom=0.1,left=0.05,right=0.95)
    ax2["A"].tick_params(axis='both',labelsize=15)
    ax2["A"].plot(dfmensualA.Month,dfmensualA.Visitantes/dfmensualA.Visitantes.sum(),color="black")
    ax2["A"].plot(dfmensualA.Month,dfmensualA.turistas/dfmensualA.turistas.sum(),color="red")
    ax2["A"].plot(dfmensualA.Month,dfmensualA.PUD/dfmensualA.PUD.sum(),color="blue")
    ax2["A"].set_title(str(dfmensualA.IdOAPN.iloc[0]),fontsize=15)
    #ax2["A"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2["B"].tick_params(axis='both',labelsize=15)
    ax2["B"].plot(dfmensualB.Month,dfmensualB.Visitantes/dfmensualB.Visitantes.sum(),color="black")
    ax2["B"].plot(dfmensualB.Month,dfmensualB.turistas/dfmensualB.turistas.sum(),color="red")
    ax2["B"].plot(dfmensualB.Month,dfmensualB.PUD/dfmensualB.PUD.sum(),color="blue")
    ax2["B"].set_title(str(dfmensualB.IdOAPN.iloc[0]),fontsize=15)
    # ax2["B"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
 
    ax2["C"].tick_params(axis='both',labelsize=15)
    ax2["C"].plot(dfmensualC.Month,dfmensualC.Visitantes/dfmensualC.Visitantes.sum(),color="black")
    ax2["C"].plot(dfmensualC.Month,dfmensualC.turistas/dfmensualC.turistas.sum(),color="red")
    ax2["C"].plot(dfmensualC.Month,dfmensualC.PUD/dfmensualC.PUD.sum(),color="blue")
    ax2["C"].set_title(str(dfmensualC.IdOAPN.iloc[0]),fontsize=15)
    # ax2["C"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    
    ax2["D"].tick_params(axis='both',labelsize=15)
    ax2["D"].plot(dfmensualD.Month,dfmensualD.Visitantes/dfmensualD.Visitantes.sum(),color="black")
    ax2["D"].plot(dfmensualD.Month,dfmensualD.turistas/dfmensualD.turistas.sum(),color="red")
    ax2["D"].plot(dfmensualD.Month,dfmensualD.PUD/dfmensualD.PUD.sum(),color="blue")
    ax2["D"].set_title(str(dfmensualD.IdOAPN.iloc[0]),fontsize=15)
    # ax2["D"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    
    ax2["E"].tick_params(axis='both',labelsize=15)
    ax2["E"].plot(dfmensualE.Month,dfmensualE.Visitantes/dfmensualE.Visitantes.sum(),color="black")
    ax2["E"].plot(dfmensualE.Month,dfmensualE.turistas/dfmensualE.turistas.sum(),color="red")
    ax2["E"].plot(dfmensualE.Month,dfmensualE.PUD/dfmensualE.PUD.sum(),color="blue")
    ax2["E"].set_title(str(dfmensualE.IdOAPN.iloc[0]),fontsize=15)
    # ax2["E"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    
    ax2["F"].tick_params(axis='both',labelsize=15)
    ax2["F"].plot(dfmensualF.Month,dfmensualF.Visitantes/dfmensualF.Visitantes.sum(),color="black")
    ax2["F"].plot(dfmensualF.Month,dfmensualF.turistas/dfmensualF.turistas.sum(),color="red")
    ax2["F"].plot(dfmensualF.Month,dfmensualF.PUD/dfmensualF.PUD.sum(),color="blue")
    ax2["F"].set_title(str(dfmensualF.IdOAPN.iloc[0]),fontsize=15)
    # ax2["F"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
 
    ax2["G"].tick_params(axis='both',labelsize=15)
    ax2["G"].plot(dfmensualG.Month,dfmensualG.Visitantes/dfmensualG.Visitantes.sum(),color="black")
    ax2["G"].plot(dfmensualG.Month,dfmensualG.turistas/dfmensualG.turistas.sum(),color="red")
    ax2["G"].plot(dfmensualG.Month,dfmensualG.PUD/dfmensualG.PUD.sum(),color="blue")
    ax2["G"].plot(dfmensualG.Month,dfmensualG.IUD/dfmensualG.IUD.sum(),color="purple")
    ax2["G"].set_title(str(dfmensualG.IdOAPN.iloc[0]),fontsize=15)
    # ax2["G"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
 
    ax2["H"].tick_params(axis='both',labelsize=15)
    ax2["H"].plot(dfmensualH.Month,dfmensualH.Visitantes/dfmensualH.Visitantes.sum(),color="black")
    ax2["H"].plot(dfmensualH.Month,dfmensualH.turistas/dfmensualH.turistas.sum(),color="red")
    ax2["H"].plot(dfmensualH.Month,dfmensualH.PUD/dfmensualH.PUD.sum(),color="blue")
    ax2["H"].set_title(str(dfmensualH.IdOAPN.iloc[0]),fontsize=15)
    # ax2["H"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
 
    ax2["I"].tick_params(axis='both',labelsize=15)
    ax2["I"].plot(dfmensualI.Month,dfmensualI.Visitantes/dfmensualI.Visitantes.sum(),color="black")
    ax2["I"].plot(dfmensualI.Month,dfmensualI.turistas/dfmensualI.turistas.sum(),color="red")
    ax2["I"].plot(dfmensualI.Month,dfmensualI.PUD/dfmensualI.PUD.sum(),color="blue")
    ax2["I"].set_title(str(dfmensualI.IdOAPN.iloc[0]),fontsize=15)
    # ax2["I"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
 
    ax2["J"].tick_params(axis='both',labelsize=15)
    ax2["J"].plot(dfmensualJ.Month,dfmensualJ.Visitantes/dfmensualJ.Visitantes.sum(),color="black")
    ax2["J"].plot(dfmensualJ.Month,dfmensualJ.turistas/dfmensualJ.turistas.sum(),color="red")
    ax2["J"].plot(dfmensualJ.Month,dfmensualJ.PUD/dfmensualJ.PUD.sum(),color="blue")
    ax2["J"].set_title(str(dfmensualJ.IdOAPN.iloc[0]),fontsize=15)
    # ax2["J"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
 
    ax2["K"].tick_params(axis='both',labelsize=15)
    ax2["K"].plot(dfmensualK.Month,dfmensualK.Visitantes/dfmensualK.Visitantes.sum(),color="black")
    ax2["K"].plot(dfmensualK.Month,dfmensualK.turistas/dfmensualK.turistas.sum(),color="red")
    ax2["K"].plot(dfmensualK.Month,dfmensualK.PUD/dfmensualK.PUD.sum(),color="blue")
    ax2["K"].set_title(str(dfmensualK.IdOAPN.iloc[0]),fontsize=15)
    # ax2["K"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
 
    ax2["L"].tick_params(axis='both',labelsize=15)
    ax2["L"].plot(dfmensualL.Month,dfmensualL.Visitantes/dfmensualL.Visitantes.sum(),color="black")
    ax2["L"].plot(dfmensualL.Month,dfmensualL.turistas/dfmensualL.turistas.sum(),color="red")
    ax2["L"].plot(dfmensualL.Month,dfmensualL.PUD/dfmensualL.PUD.sum(),color="blue")
    ax2["L"].set_title(str(dfmensualL.IdOAPN.iloc[0]),fontsize=15)
    # ax2["L"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
 
    ax2["M"].tick_params(axis='both',labelsize=15)
    ax2["M"].plot(dfmensualM.Month,dfmensualM.Visitantes/dfmensualM.Visitantes.sum(),color="black")
    ax2["M"].plot(dfmensualM.Month,dfmensualM.turistas/dfmensualM.turistas.sum(),color="red")
    ax2["M"].plot(dfmensualM.Month,dfmensualM.PUD/dfmensualM.PUD.sum(),color="blue")
    ax2["M"].set_title(str(dfmensualM.IdOAPN.iloc[0]),fontsize=15)
    # ax2["M"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2["N"].tick_params(axis='both',labelsize=15)
    ax2["N"].plot(dfmensualN.Month,dfmensualN.Visitantes/dfmensualN.Visitantes.sum(),color="black")
    ax2["N"].plot(dfmensualN.Month,dfmensualN.turistas/dfmensualN.turistas.sum(),color="red")
    ax2["N"].plot(dfmensualN.Month,dfmensualN.PUD/dfmensualN.PUD.sum(),color="blue")
    ax2["N"].plot(dfmensualN.Month,dfmensualN.IUD/dfmensualN.IUD.sum(),color="purple")
    ax2["N"].set_title(str(dfmensualN.IdOAPN.iloc[0]),fontsize=15)
    # ax2["N"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2["O"].tick_params(axis='both',labelsize=15)
    ax2["O"].plot(dfmensualO.Month,dfmensualO.Visitantes/dfmensualO.Visitantes.sum(),color="black")
    ax2["O"].plot(dfmensualO.Month,dfmensualO.turistas/dfmensualO.turistas.sum(),color="red")
    ax2["O"].plot(dfmensualO.Month,dfmensualO.PUD/dfmensualO.PUD.sum(),color="blue")
    ax2["O"].set_title(str(dfmensualO.IdOAPN.iloc[0]),fontsize=15)
    # ax2["O"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2["P"].tick_params(axis='both',labelsize=15)
    ax2["P"].plot(dfmensualP.Month,dfmensualP.Visitantes/dfmensualP.Visitantes.sum(),color="black")
    ax2["P"].plot(dfmensualP.Month,dfmensualP.turistas/dfmensualP.turistas.sum(),color="red")
    ax2["P"].plot(dfmensualP.Month,dfmensualP.PUD/dfmensualP.PUD.sum(),color="blue")
    ax2["P"].plot(dfmensualP.Month,dfmensualP.IUD/dfmensualP.IUD.sum(),color="purple")
    ax2["P"].set_title(str(dfmensualP.IdOAPN.iloc[0]),fontsize=15)
    # ax2["P"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    #fig2.savefig("/home/usuario/Documentos/recreation/imagenes_paper.pdf")
    plt.show()
