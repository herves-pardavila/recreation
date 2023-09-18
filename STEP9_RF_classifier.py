import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
sys.path.append("/home/usuario/OneDrive/MEXILLÓN_nueva/2stage_model")
from random_forest_classifier import train_random_forest,r2,tune_mse,variable_importance

if __name__ =="__main__":

    df=pd.read_csv("PUD.csv")
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel("Photo-user-days")
    ax.title.set_text("Histograma")
    df.PUD.hist(ax=ax)
    plt.show()

    df.loc[df.PUD >0,"PUD"]=1
    explanatory_variables=set(df.columns)
    explanatory_variables.remove("PUD")
    explanatory_variables.remove("FID")
    explanatory_variables.remove("date")
    explanatory_variables=list(explanatory_variables)
    #explanatory_variables=["TA_MAX_1.5m","HSOL_SUM_1.5m","PRED_AVG_1.5m","VV_AVG_2m","TA_AVG_1.5m","TA_MIN_1.5m","Month","restaurant","PP_SUM_1.5m"]
    variable_y="PUD"

    df=df[explanatory_variables+["PUD"]]
    df.dropna(inplace=True,subset=explanatory_variables+[variable_y])

    print("Tamaño df=",len(df))

    # #MSE tunning
    # mse=tune_mse(df,explanatory_variables,"PUD")
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.plot(explanatory_variables,mse)
    # plt.show()
    
    # # Tunning of D and T

    # Ds=[10,20,30,40,50]
    # Ts=[50,60,70,80,90]
    # xx=np.zeros((len(Ds),len(Ts)))
    # for i in range(len(Ds)):
    #     print(Ds[i])
    #     for j in range(len(Ts)):
    #         print(Ts[j])
    #         modelo,train,test,rsq=train_random_forest(df,Ts[j],rho=len(explanatory_variables)/3,D=Ds[i],training=0.7,variables=explanatory_variables,y_variable="PUD",bootstrap=0.9)
    #         conf_mat = confusion_matrix(test[variable_y], test["yhat"],normalize="true")
    #         xx[i,j]=conf_mat[1][1]
    # print(xx)

    modelo,train,test,rsq=train_random_forest(df,T=50,rho=len(explanatory_variables)/3,D=20,training=0.7,variables=explanatory_variables,y_variable="PUD",bootstrap=0.9)
    conf_mat = confusion_matrix(test[variable_y], test["yhat"],normalize="true")
    print(conf_mat)
    #importance=variable_importance(df,explanatory_variables,variable_y)
    #print(importance/importance.loc["random"]["Importance"])


    #PartialDependenceDisplay.from_estimator(modelo, train, features=["Month"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    PartialDependenceDisplay.from_estimator(modelo, train, features=["TA_MAX_1.5m"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["PP_SUM_1.5m"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["VV_AVG_2m"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["HSOL_SUM_1.5m"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["PRED_AVG_1.5m"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["TA_AVG_1.5m"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["TA_MIN_1.5m"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["Month"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["restaurant"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["PP_SUM_1.5m"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))

    plt.show()

    
