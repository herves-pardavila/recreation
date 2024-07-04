import pandas as pd
import pypopulation
import numpy as np
from datetime import datetime
if __name__ == "__main__":

    df=pd.read_csv("tourist_origins_distances.csv")

    #add population of countries
    print(df[pd.isna(df.Poblacion)]["origen"].unique())
    for code in df.CNTR_ID.unique():
        print(code)
        try:
            df.loc[df.CNTR_ID==code,"Poblacion"]=pypopulation.get_population(code)
        except AttributeError: 
            continue
    
    
    
    
    df.loc[df.origen=="Reino Unido","Poblacion"]=66834405
    df.loc[df.origen=="Grecia","Poblacion"]=10716322
    print(df[pd.isna(df.Poblacion)]["CNTR_ID"].unique())
    print(df[df.CNTR_ID=="UK"])
    # #add other important columns
    # df.Date=pd.to_datetime(df.Date)
    # df["Month"]=df.Date.dt.month
    # df["Year"]=df.Date.dt.year
    # df.loc[df.Month.isin([1,2,3]),"Season"]="Winter"
    # df.loc[df.Month.isin([3,4,5]),"Season"]="Spring"
    # df.loc[df.Month.isin([6,7,8,9]),"Season"]="Summer"
    # df.loc[df.Month.isin([10,11,12]),"Season"]="Fall"

    # #add covid
    # df["covid"]=0
    # df.loc[((df.Date > datetime(2020,2,1)) & (df.Date < datetime(2021,6,1))),"covid"]=1
    # print(df.covid.unique())
    # df=df[~((df.Date > datetime(2020,2,1)) & (df.Date < datetime(2020,8,1)))]
    # print(df)

    # df.to_csv("travel_cost.csv",index=False)


    


