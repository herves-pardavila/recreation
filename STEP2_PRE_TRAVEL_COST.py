import pandas as pd
# import numpy as np
# from patsy import dmatrices
# import statsmodels.api as sm
import matplotlib.pyplot as plt
# import statsmodels.formula.api as smf
# import statsmodels.graphics as smg
# from sklearn.metrics import r2_score
# from scipy import stats
# from datetime import datetime

if __name__== "__main__":
    path=r"F:/doctorado/"
    #load the data
    df=pd.read_csv(path+"recreation/ZonalTravelCost/datos_provinciales/3travel_cost_Ons.csv")
    #df=df[df.Año==2019]
    print(df)
    
    df["distance"]=df["distance (km)"].astype(float)
    
    df["median_inc"]=df.Median_I/365
    df["CT_(€)"]=0
    
    #cost of travel
    df["CT_(€)"]=df.distance*0.26
   

    #opoprtunity cost
    df["OC_(€)"]=(1/3)*df.median_inc

   
    print(df)
    print(df.info())
    df["TC"]=df["CT_(€)"]+df["OC_(€)"]
    df.to_csv(path+"recreation/ZonalTravelCost/datos_provinciales/3travel_cost_Ons_ready.csv",index=False)


    #====== FALTAN UNIR LOS TURISTAS INTERNACIONALES========================


    fig=plt.figure()
    fig.suptitle("Demand Curve")
    ax=fig.add_subplot(111)
    ax.set_ylabel("Trip Cost (p)")
    ax.set_xlabel("Trips (Q)")
    ax.plot(df.turistasINE,df["CT_(€)"]+df["OC_(€)"],"o",color="red",label="INE Data")
    ax.set_xlim(0,10e4)
    fig.legend() 
    plt.show()