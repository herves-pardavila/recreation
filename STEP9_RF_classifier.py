import sklearn as skl
import pandas as pd
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import datetime
from rfpimp import permutation_importances
from sklearn.metrics import r2_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix
import geopandas as gpd
import contextily as ctx
import sys
sys.path.append("/home/usuario/OneDrive/MEXILLÓN_nueva/2stage_model")
from random_forest_classifier import tune_mse


def train_random_forest(data,T,rho,D,training,variables,y_variable,bootstrap):
    train_set,test_set=skl.model_selection.train_test_split(data,train_size=training,random_state=1)
    rf=ensemble.RandomForestClassifier(n_estimators=T,criterion="entropy",max_depth=D,max_features="sqrt",class_weight="balanced",oob_score=True,random_state=10,max_samples=bootstrap)
    
    X_train=train_set[variables]
    y_train=train_set[y_variable]
    
    X_test=test_set[variables]
    y_test=test_set[y_variable]
    
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    rsq=rf.score(X_test,y_test)

    test_set["yhat"]=y_pred
    
    return rf,X_train,y_train,test_set,rsq

def score(modelo,X_train, y_train):
    return confusion_matrix(y_train, modelo.predict(X_train),normalize="true")[1][1]

def variable_importance(df,variables,variable_dependiente):
    df["random"]=np.random.random(size=len(df))
    train_set,test_set=skl.model_selection.train_test_split(df,train_size=0.7,random_state=1)
    variables=variables+["random"]
    modelo,X_train,y_train,test,rsq=train_random_forest(df,T=50,rho=len(variables)/3,D=20,training=0.7,variables=variables,y_variable=variable_dependiente,bootstrap=0.9)
    X_train=train_set[variables]
    y_train=train_set[variable_dependiente]   
    modelo.fit(X_train,y_train)
    perm_imp_rfpimp = permutation_importances(modelo, X_train, y_train, score)
    return perm_imp_rfpimp

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
    explanatory_variables.remove("camp_pitch")
    explanatory_variables.remove("camp_site")
    explanatory_variables.remove("motel")
    explanatory_variables.remove("caravan_site")

    explanatory_variables=list(explanatory_variables)
    print(explanatory_variables)
    explanatory_variables=["TA_MAX_1.5m","HSOL_SUM_1.5m","PRED_AVG_1.5m","VV_AVG_2m","TA_AVG_1.5m","TA_MIN_1.5m","Month","PP_SUM_1.5m","FID"]
    variable_y="PUD"

    df=df[explanatory_variables+["PUD"]]
    df.dropna(inplace=True,subset=explanatory_variables+[variable_y])
    
    print("Tamaño df=",len(df))

   
    modelo,X_train,y_train,test,rsq=train_random_forest(df,T=50,rho=len(explanatory_variables)/3,D=20,training=0.7,variables=explanatory_variables,y_variable="PUD",bootstrap=0.9)
    conf_mat = confusion_matrix(test[variable_y], test["yhat"],normalize="true")
    print(conf_mat)
    importance=variable_importance(df,explanatory_variables,variable_y)
    print(importance)
    print(importance/importance.loc["random"]["Importance"])


    PartialDependenceDisplay.from_estimator(modelo, X_train, features=["Month"],target=1,method="brute",categorical_features=["Month","FID"],percentiles=(0.1,0.9))
    PartialDependenceDisplay.from_estimator(modelo, X_train, features=["TA_MAX_1.5m"],target=1,method="brute",categorical_features=["Month","FID"],percentiles=(0.1,0.9))
    PartialDependenceDisplay.from_estimator(modelo, X_train, features=["PP_SUM_1.5m"],target=1,method="brute",categorical_features=["Month","FID"],percentiles=(0.1,0.9))
    PartialDependenceDisplay.from_estimator(modelo, X_train, features=["VV_AVG_2m"],target=1,method="brute",categorical_features=["Month","FID"],percentiles=(0.1,0.9))
    PartialDependenceDisplay.from_estimator(modelo, X_train, features=["HSOL_SUM_1.5m"],target=1,method="brute",categorical_features=["Month","FID"],percentiles=(0.1,0.9))
    PartialDependenceDisplay.from_estimator(modelo, X_train, features=["PRED_AVG_1.5m"],target=1,method="brute",categorical_features=["Month","FID"],percentiles=(0.1,0.9))
    PartialDependenceDisplay.from_estimator(modelo, X_train, features=["TA_AVG_1.5m"],target=1,method="brute",categorical_features=["Month","FID"],percentiles=(0.1,0.9))
    PartialDependenceDisplay.from_estimator(modelo, X_train, features=["TA_MIN_1.5m"],target=1,method="brute",categorical_features=["Month","FID"],percentiles=(0.1,0.9))
    #PartialDependenceDisplay.from_estimator(modelo, X_train, features=["FID"],target=1,method="brute",categorical_features=["Month","FID"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["restaurant"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))
    # PartialDependenceDisplay.from_estimator(modelo, train, features=["PP_SUM_1.5m"],target=1,method="brute",categorical_features=["Month"],percentiles=(0.1,0.9))

    plt.show()


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

    
