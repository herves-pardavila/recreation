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
    path=r"D:/doctorado/"
    #load the data
    df=pd.read_csv(path+"recreation/ZonalTravelCost/datos_municipales/3travel_cost_Ons_with_Income.csv")
    #df=df[df.Año==2019]
    print(df)
    
    df["distance"]=df["distance (km)"].astype(float)
    
    df.RBMPP=df.RBMPP/365
    
    df["CT_(€)"]=0
    
    #cost of travel
    df.loc[df.Zona.isin(["Resto","Galicia"]),"CT_(€)"]=df.distance*0.26
    df.loc[df.Zona.isin(["Isleños"]),"CT_(€)"]=df.distance*0.25*0.15
   

    #opoprtunity cost
    df["OC_(€)"]=(2/3)*df.RBMPP
    df.loc[df.Zona=="Galicia","OC_(€)"]=(1/3)*df.RBMPP

   
    print(df)
    print(df.info())
    df["TC"]=df["CT_(€)"]+df["OC_(€)"]
    


    #====== FALTAN UNIR LOS TURISTAS INTERNACIONALES========================
    df_externo=pd.read_csv(path + "recreation/ZonalTravelCost/3travel_cost_Ons_ready.csv")
    df_externo = df_externo[df_externo.Zona != "España"]
    
    newdf = pd.concat([df,df_externo],axis=0,join="outer",ignore_index=False)
    
    newdf.to_csv(path+"recreation/ZonalTravelCost/datos_municipales/3travel_cost_Ons_ready.csv",index=False)

    fig=plt.figure()
    fig.suptitle("Demand Curve")
    ax=fig.add_subplot(111)
    ax.set_ylabel("Trip Cost (p)")
    ax.set_xlabel("Trips (Q)")
    ax.plot(df.turistasINE,df["TC"],"o",color="red",label="INE Data")
    ax.set_xlim(0,10e4)
    fig.legend() 
    plt.show()