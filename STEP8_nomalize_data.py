import pandas as pd

if __name__== "__main__":

    pud=pd.read_csv("PUD.csv")
    print(pud.columns)
    columns=set(pud.columns)
    columns.remove("FID")
    columns.remove("PUD")
    columns.remove("date")
    print(columns)
    for variable in columns:
        pud[variable]=1+(pud[variable]-pud[variable].mean())/pud[variable].mean()
    
    print(pud.describe())
    pud.date=pd.to_datetime(pud.date)
    pud["Month"]=pud.date.dt.month
    pud.to_csv("PUD.csv",index=False)