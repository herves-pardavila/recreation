import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import numpy as np
import time

def scrapper(hastags):

    driver=webdriver.Firefox()
    for i in range(len(hastags)):
        print(hastags.iloc[i]["Location"])
        try:
            driver.get("https://www.google.com/search?q=%s+explore+locations+instagram" %str(hastags.iloc[i]["Location"]))
            driver.find_element(By.CSS_SELECTOR,"#W0wltc > div:nth-child(1)").click()
            driver.find_element(By.CSS_SELECTOR,"#rso > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > span:nth-child(1) > a:nth-child(1) > h3:nth-child(2)").click()
            url=driver.current_url
            hastags.loc[i]["url"]=url
            time.sleep(10)
        except:
            try:
                driver.find_element(By.CSS_SELECTOR,"#rso > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > span:nth-child(1) > a:nth-child(1) > h3:nth-child(2)").click()
                url=driver.current_url
                hastags.loc[i]["url"]=url
                time.sleep(10)
            except:
                time.sleep(10)    
                continue

        
        
    return hastags

if __name__ == "__main__":


    df=pd.read_csv("new_df.csv")
    df=scrapper(df)
    fun= lambda x : [int(s) for s in x.split("/") if s.isdigit()]
    lista=list(map(fun,df.url))
    replace= lambda y: str(y).replace("[","").replace("]","")
    newlist=list(map(replace,lista))
    df["codes"]=newlist
    df.to_csv("scrapping_locations2.csv",index=False)
    subdf=df[df.codes!=""]
    with open('location_codes2.txt', 'w') as f:
        for line in subdf.codes:
            f.write("%s\n" % line)

    
