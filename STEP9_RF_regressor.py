import geopandas as gpd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import sys
sys.path.append("/home/usuario/OneDrive/MEXILLÓN_nueva/2stage_model")
from random_forest_regressor import train_random_forest,r2,tune_mse,variable_importance

if __name__ =="__main__":
    gdf=gpd.read_file("PUD.shp")
    independent_variables=["dist-train","trial_len", "hotels", "Com-nodes","protected",
                        "dem","HSOL_SUM_1","PRED_AVG_1","TA_AVG_1.5","TA_MAX_1.5","TA_MIN_1.5","PP_SUM_1.5","VV_AVG_2m",
                        "viewpoint","beach","restaurant","bike_len","nautic","lighthouse",
                        "town (m)","city (m)","arable","built","crops","forest","grassland","intertidal",
                        "marshes","rocks","scrub"]
    variable_y="PUD"
    gdf=gdf[independent_variables+["PUD"]]
    #gdf=gdf.dropna(inplace=True)
    print("Tamaño df=",len(gdf))

    #mse tunning
    mse=tune_mse(gdf,independent_variables,"PUD")
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(independent_variables,mse)
    plt.show()
    
    
    modelo,train,test,rsq=train_random_forest(gdf,6000,rho=len(independent_variables)/3,D=30,training=0.7,variables=independent_variables,y_variable="PUD",bootstrap=0.9)
    mse=metrics.mean_squared_error(test["PUD"],test["yhat"])
    print("MSE=",mse)

    fig2=plt.figure()
    ax2=fig2.add_subplot(111)
    ax2.plot(np.arange(1,len(test[variable_y])+1,1),test[variable_y],label="test data")
    ax2.plot(np.arange(1,len(test["yhat"])+1,1),test["yhat"],label="predicted data")
    fig2.suptitle("Rsquared=%f"%rsq)
    fig2.legend()
    plt.show()

    importance=variable_importance(gdf,independent_variables,variable_y)
    print(importance/importance.loc["random"]["Importance"])
    
    #Partial dependence plots
   
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.set_xlabel("Dias de Cierre ")
    # PartialDependenceDisplay.from_estimator(modelo, train, features=[15],ax=ax)
    # ax.title.set_text("Dias de Cierre ")
    # plt.show()