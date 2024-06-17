import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics as smg
from sklearn.metrics import r2_score
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
if __name__== "__main__":

    #prepare the data
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation_ready.csv")
    df=df[df.IdOAPN=="Timanfaya"]
    df.Date=df.Date.astype("category")
    df.Month=df.Month.astype("category")
    df.Year=df.Year.astype("category")
    df.IdOAPN=df.IdOAPN.astype("category")
    df.Season=df.Season.astype("category")
    df.covid=df.covid.astype("category")

    df=df[["Date","IdOAPN","logPUD","Visitantes","Season","covid"]]

    #divide between training set and test set
    df.dropna(subset=["Visitantes","logPUD"],inplace=True)
    df.Visitantes=df.Visitantes.astype(int)
    #df.turistas_total=df.turistas_total.astype(int)

  
    for park in df.IdOAPN.unique():
        newdf=df[df.IdOAPN==park]
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(newdf.logPUD,newdf.Visitantes,"*",label=str(park))
        fig.legend()
        plt.show()

    #SVR
    expr1="""Visitantes ~ logPUD + Season + covid"""
    expr2="""Visitantes ~ turistas_total+ IdOAPN + Season + covid"""
    expr5="""Visitantes ~ logPUD + IdOAPN + Season + covid"""
    null_expr="""Visitantes ~ IdOAPN + Season + covid """
    expr=expr1
    np.random.seed(seed=1)
    mask=np.random.rand(len(df))< 0.7
    df_train=df[mask]  
    df_test=df[~mask]
    y_train, X_train = dmatrices(expr, df_train, return_type='matrix')
    y_test, X_test = dmatrices(expr, df_test, return_type='matrix')


    clf = SVR(kernel='rbf',epsilon=0.1,gamma="auto", C=100,tol=10)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


