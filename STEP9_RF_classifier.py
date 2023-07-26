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

    # #mse tunning
    # mse=tune_mse(df,explanatory_variables,"PUD")
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.plot(explanatory_variables,mse)
    # plt.show()
    
    
    modelo,train,test,rsq=train_random_forest(df,1000,rho=len(explanatory_variables)/3,D=30,training=0.7,variables=explanatory_variables,y_variable="PUD",bootstrap=0.9)
    mse=metrics.mean_squared_error(test["PUD"],test["yhat"])
    print("MSE=",mse)

    fig2=plt.figure()
    ax2=fig2.add_subplot(111)
    ax2.plot(np.arange(1,len(test[variable_y])+1,1),test[variable_y],label="test data")
    ax2.plot(np.arange(1,len(test["yhat"])+1,1),test["yhat"],label="predicted data")
    fig2.suptitle("Rsquared=%f"%rsq)
    fig2.legend()
    plt.show()

    importance=variable_importance(df,explanatory_variables,variable_y)
    print(importance/importance.loc["random"]["Importance"])

    print(test.yhat.unique())
    print(test[variable_y].unique())
    conf_mat = confusion_matrix(test[variable_y], test["yhat"],normalize="true")
    print(conf_mat)
    
    #Partial dependence plots
   
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.set_xlabel("Dias de Cierre ")
    # PartialDependenceDisplay.from_estimator(modelo, train, features=[15],ax=ax)
    # ax.title.set_text("Dias de Cierre ")
    # plt.show()