import pandas as pd
import pypopulation
if __name__ == "__main__":

    df=pd.read_csv("tourist_origins_distances.csv")
    print(df[pd.isna(df.Poblacion)]["origen"].unique())
    for code in df.CNTR_ID.unique():
        print(code)
        try:
            df.loc[df.CNTR_ID==code,"Poblacion"]=pypopulation.get_population(code)
        except AttributeError: 
            continue
    print(df[pd.isna(df.Poblacion)]["origen"].unique())
    df.loc[df.origen=="Reino Unido","Poblacion"]=66834405
    df.loc[df.origen=="Grecia","Poblacion"]=10716322
    df.to_csv("travel_cost.csv",index=False)

    


    


