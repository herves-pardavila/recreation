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


def CT_Avion(dataframe):
    
    for i in range(len(dataframe)):
        print(i)
        if pd.isna(dataframe.loc[i, "Distancia Avion (km) Ons"]):
            continue
        elif dataframe.iloc[i][["Distancia Avion (km) Ons"]].between(801,1200):
            dataframe.loc[i,"CT_(€)"]=0.138
        elif dataframe.iloc[i][["Distancia Avion (km) Ons"]].between(1601,2000):
            dataframe.loc[i, "CT_(€)"]=0.1025
        else:
            print("Caso no comtemplado en la función CT_Avion, REVISAR!")
    return dataframe

if __name__== "__main__":
    path=r"D:/doctorado/"
    #load the data
    df=pd.read_csv(path+"recreation/ZonalTravelCost/datos_provinciales/3travel_cost_Ons.csv")
    #df=df[df.Año==2019]
    print(df)
    
    df["distance"]=df["distance (km)"].astype(float)
    
    df["RBMPP"]=df.RBMPP/365
    df["CT_(€)"]=0
    
    #cost of travel
    df["CT_(€)"]=df.distance*0.26
   

    #opoprtunity cost
    df["OC_(€)"]=(1/3)*df.RBMPP

   
    print(df)
    print(df.info())
    df["TC"]=df["CT_(€)"]+df["OC_(€)"]
    


    #====== FALTAN UNIR LOS TURISTAS INTERNACIONALES========================
    df_externo=pd.read_csv(path + "recreation/ZonalTravelCost/3travel_cost_Ons_ready.csv")
    df_externo = df_externo[df_externo.Zona != "España"]
    
    newdf = pd.concat([df,df_externo],axis=0,join="inner",ignore_index=False)
    
    newdf.to_csv(path+"recreation/ZonalTravelCost/datos_provinciales/3travel_cost_Ons_ready.csv",index=False)

    fig=plt.figure()
    fig.suptitle("Demand Curve")
    ax=fig.add_subplot(111)
    ax.set_ylabel("Trip Cost (p)")
    ax.set_xlabel("Trips (Q)")
    ax.plot(newdf.turistasINE,newdf["TC"],"o",color="red",label="INE Data")
    ax.set_xlim(0,10e4)
    fig.legend() 
    plt.show()