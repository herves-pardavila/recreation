import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
from sklearn.metrics import r2_score
from scipy import stats
from datetime import datetime

if __name__== "__main__":
    
    #load the data
    df=pd.read_csv("3travel_cost.csv")
    print(df)
    
    df["distance"]=df["distance (km)"].astype(float)
    df.loc[df.Lugar=="Andorra","median_inc"]=41640
    df.median_inc=df.median_inc/365
    df["CT_(€)"]=0
    
    #cost of travel
    df.loc[df.Zona=="España","CT_(€)"]=df.distance*0.26/2
    df.loc[df.Lugar.isin(["Portugal","Andorra"]),"CT_(€)"]=df.distance*0.26
    df.loc[df.Lugar.isin(["Francia","Reino Unido","Italia","Suiza","Bélgica","Irlanda","Andorra","Países Bajos"]),"CT_(€)"]=df.distance*0.1*2
    df.loc[df.Lugar.isin(["Austria","Polonia","Brasil","Alemania","Estados Unidos","República Checa"]),"CT_(€)"]=df.distance*0.06*2

    #opoprtunity cost
    df["OC_(€)"]=2*(1/3)*df.median_inc
    df.loc[df.Zona=="España","OC_(€)"]=(1/3)*df.median_inc
    df.loc[df.Lugar.isin(["Portugal","Andorra"]),"OC_(€)"]=(1/3)*df.median_inc
   
    print(df)
    print(df.info())
    df["TC"]=df["CT_(€)"]+df["OC_(€)"]
    df.to_csv("3travel_cost_ready.csv",index=False)

    fig=plt.figure()
    fig.suptitle("Demand Curve")
    ax=fig.add_subplot(111)
    ax.set_ylabel("Trip Cost")
    ax.set_xlabel("Trips per season")
    ax.plot(df.Numero,df["CT_(€)"]+df["OC_(€)"],"o",color="black",label="Real Data")
    ax.plot(df.turistasINE,df["CT_(€)"]+df["OC_(€)"],"o",color="red",label="INE Data")
    ax.plot(df.yhat_full,df["CT_(€)"]+df["OC_(€)"],"o",color="blue",label="Predictor")
            
    plt.show()