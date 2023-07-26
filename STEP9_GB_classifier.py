import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
import sys
sys.path.append("/home/usuario/OneDrive/MEXILLÓN_nueva/2stage_model")
from gradient_boost_classifier import train_gradient_boost_classifier
from random_forest_classifier import train_random_forest,r2,tune_mse,variable_importance


if __name__ =="__main__":

    df=pd.read_csv("PUD.csv")
    df.loc[df.PUD >0,"PUD"]=1
    explanatory_variables=set(df.columns)
    explanatory_variables.remove("PUD")
    explanatory_variables.remove("FID")
    explanatory_variables.remove("date")
    explanatory_variables=list(explanatory_variables)
    variable_y="PUD"

    df=df[explanatory_variables+["PUD"]]
    df.dropna(inplace=True,subset=explanatory_variables+[variable_y])

    print("Tamaño df=",len(df))

    Ds=[50]
    Ms=[100,250,500]
    xx=np.zeros((len(Ds),len(Ms)))
    for i in range(len(Ds)):
        print(Ds[i])
        for j in range(len(Ms)):
            print(Ms[j])
            modelo,train,test=train_gradient_boost_classifier(data=df,M=Ms[j],D=Ds[i],training=0.7,variables=explanatory_variables,y_variable="PUD")
            conf_mat = confusion_matrix(test[variable_y], test["yhat"],normalize="true")
            xx[i,j]=conf_mat[1][1]
    print(xx)        

