import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__== "__main__":

    #anual visitors
    df=pd.read_csv("/home/usuario/Documentos/recreation/recreation.csv")
    print(df.Month.unique())
    dfanual=df.groupby(by=["SITE_NAME"],as_index=False).sum(numeric_only=True)
    print(dfanual.Month.unique())
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.loglog(dfanual.Visitantes,dfanual.PUD,"*",label="Flickr PUD")
    ax.loglog(dfanual.Visitantes,dfanual.turistas_total,"*",label="INE data")
    ax.loglog(dfanual.Visitantes,dfanual.turistas_corregido,"*",label="INE data corregido")
    ax.loglog(np.arange(1e5,8e7,10),np.arange(1e5,8e7,10),label="1-1 line")
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.PUD,dfanual.SITE_NAME)]
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.turistas_total,dfanual.SITE_NAME)]
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.turistas_corregido,dfanual.SITE_NAME)]
    ax.set_xlabel("Visitors")
    ax.set_ylabel("Estimated User-generated-data visitors")
    fig.legend()
    plt.show()

    #Monthly visitation per park
    dfmensual=df.groupby(by=["SITE_NAME","Month"],as_index=False).sum(numeric_only=True)
    print(dfmensual.Month.unique())
    print(dfmensual)
    dfmensualA=dfmensual[dfmensual.SITE_NAME=="Aigüestortes i Estany de Sant Maurici"]
    dfmensualB=dfmensual[dfmensual.SITE_NAME=="Archipiélago de Cabrera"]
    dfmensualC=dfmensual[dfmensual.SITE_NAME=="Cabañeros"]
    dfmensualD=dfmensual[dfmensual.SITE_NAME=="Caldera de Taburiente"]
    dfmensualE=dfmensual[dfmensual.SITE_NAME=="Doñana"]
    dfmensualF=dfmensual[dfmensual.SITE_NAME=="Garajonay"]
    dfmensualG=dfmensual[dfmensual.SITE_NAME=="Islas Atlánticas de Galicia"]
    dfmensualH=dfmensual[dfmensual.SITE_NAME=="Monfragüe"]
    dfmensualI=dfmensual[dfmensual.SITE_NAME=="Ordesa y Monte Perdido"]
    dfmensualJ=dfmensual[dfmensual.SITE_NAME=="Picos de Europa"]
    dfmensualK=dfmensual[dfmensual.SITE_NAME=="Sierra Nevada"]
    print(dfmensualK)
    dfmensualL=dfmensual[dfmensual.SITE_NAME=="Sierra de Guadarrama"]
    dfmensualM=dfmensual[dfmensual.SITE_NAME=="Sierra de las Nieves"]
    dfmensualN=dfmensual[dfmensual.SITE_NAME=="Tablas de Daimiel"]
    dfmensualO=dfmensual[dfmensual.SITE_NAME=="Teide National Park"]
    dfmensualP=dfmensual[dfmensual.SITE_NAME=="Timanfaya"]

    fig2=plt.figure()
    ax2=fig2.subplot_mosaic("""ABCD
                         EFGH
                         IJKL
                         MNOP""")

    fig2.subplots_adjust(hspace=0.3)

    ax2["A"].plot(dfmensualA.Month,dfmensualA.Visitantes/dfmensualA.Visitantes.sum(),color="black")
    ax2["A"].plot(dfmensualA.Month,dfmensualA.turistas_corregido/dfmensualA.turistas_corregido.sum(),color="red")
    ax2["A"].plot(dfmensualA.Month,dfmensualA.PUD/dfmensualA.PUD.sum(),color="blue")
    ax2["A"].title.set_text(str(dfmensualA.SITE_NAME.unique()))
    #ax2["A"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
   
    ax2["B"].plot(dfmensualB.Month,dfmensualB.Visitantes/dfmensualB.Visitantes.sum(),color="black")
    ax2["B"].plot(dfmensualB.Month,dfmensualB.turistas_corregido/dfmensualB.turistas_corregido.sum(),color="red")
    ax2["B"].plot(dfmensualB.Month,dfmensualB.PUD/dfmensualB.PUD.sum(),color="blue")
    ax2["B"].title.set_text(str(dfmensualB.SITE_NAME.unique()))
    # ax2["B"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])


    ax2["C"].plot(dfmensualC.Month,dfmensualC.Visitantes/dfmensualC.Visitantes.sum(),color="black")
    ax2["C"].plot(dfmensualC.Month,dfmensualC.turistas_corregido/dfmensualC.turistas_corregido.sum(),color="red")
    ax2["C"].plot(dfmensualC.Month,dfmensualC.PUD/dfmensualC.PUD.sum(),color="blue")
    ax2["C"].title.set_text(str(dfmensualC.SITE_NAME.unique()))
    # ax2["C"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    

    ax2["D"].plot(dfmensualD.Month,dfmensualD.Visitantes/dfmensualD.Visitantes.sum(),color="black")
    ax2["D"].plot(dfmensualD.Month,dfmensualD.turistas_corregido/dfmensualD.turistas_corregido.sum(),color="red")
    ax2["D"].plot(dfmensualD.Month,dfmensualD.PUD/dfmensualD.PUD.sum(),color="blue")
    ax2["D"].title.set_text(str(dfmensualD.SITE_NAME.unique()))
    # ax2["D"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["E"].plot(dfmensualE.Month,dfmensualE.Visitantes/dfmensualE.Visitantes.sum(),color="black")
    ax2["E"].plot(dfmensualE.Month,dfmensualE.turistas_corregido/dfmensualE.turistas_corregido.sum(),color="red")
    ax2["E"].plot(dfmensualE.Month,dfmensualE.PUD/dfmensualE.PUD.sum(),color="blue")
    ax2["E"].title.set_text(str(dfmensualE.SITE_NAME.unique()))
    # ax2["E"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    
    ax2["F"].plot(dfmensualF.Month,dfmensualF.Visitantes/dfmensualF.Visitantes.sum(),color="black")
    ax2["F"].plot(dfmensualF.Month,dfmensualF.turistas_corregido/dfmensualF.turistas_corregido.sum(),color="red")
    ax2["F"].plot(dfmensualF.Month,dfmensualF.PUD/dfmensualF.PUD.sum(),color="blue")
    ax2["F"].title.set_text(str(dfmensualF.SITE_NAME.unique()))
    # ax2["F"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["G"].plot(dfmensualG.Month,dfmensualG.Visitantes/dfmensualG.Visitantes.sum(),color="black")
    ax2["G"].plot(dfmensualG.Month,dfmensualG.turistas_corregido/dfmensualG.turistas_corregido.sum(),color="red")
    ax2["G"].plot(dfmensualG.Month,dfmensualG.PUD/dfmensualG.PUD.sum(),color="blue")
    ax2["G"].title.set_text(str(dfmensualG.SITE_NAME.unique()))
    # ax2["G"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["H"].plot(dfmensualH.Month,dfmensualH.Visitantes/dfmensualH.Visitantes.sum(),color="black")
    ax2["H"].plot(dfmensualH.Month,dfmensualH.turistas_corregido/dfmensualH.turistas_corregido.sum(),color="red")
    ax2["H"].plot(dfmensualH.Month,dfmensualH.PUD/dfmensualH.PUD.sum(),color="blue")
    ax2["H"].title.set_text(str(dfmensualH.SITE_NAME.unique()))
    # ax2["H"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["I"].plot(dfmensualI.Month,dfmensualI.Visitantes/dfmensualI.Visitantes.sum(),color="black")
    ax2["I"].plot(dfmensualI.Month,dfmensualI.turistas_corregido/dfmensualI.turistas_corregido.sum(),color="red")
    ax2["I"].plot(dfmensualI.Month,dfmensualI.PUD/dfmensualI.PUD.sum(),color="blue")
    ax2["I"].title.set_text(str(dfmensualI.SITE_NAME.unique()))
    # ax2["I"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["J"].plot(dfmensualJ.Month,dfmensualJ.Visitantes/dfmensualJ.Visitantes.sum(),color="black")
    ax2["J"].plot(dfmensualJ.Month,dfmensualJ.turistas_corregido/dfmensualJ.turistas_corregido.sum(),color="red")
    ax2["J"].plot(dfmensualJ.Month,dfmensualJ.PUD/dfmensualJ.PUD.sum(),color="blue")
    ax2["J"].title.set_text(str(dfmensualJ.SITE_NAME.unique()))
    # ax2["J"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["K"].plot(dfmensualK.Month,dfmensualK.Visitantes/dfmensualK.Visitantes.sum(),color="black")
    ax2["K"].plot(dfmensualK.Month,dfmensualK.turistas_corregido/dfmensualK.turistas_corregido.sum(),color="red")
    ax2["K"].plot(dfmensualK.Month,dfmensualK.PUD/dfmensualK.PUD.sum(),color="blue")
    ax2["K"].title.set_text(str(dfmensualK.SITE_NAME.unique()))
    # ax2["K"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["L"].plot(dfmensualL.Month,dfmensualL.Visitantes/dfmensualL.Visitantes.sum(),color="black")
    ax2["L"].plot(dfmensualL.Month,dfmensualL.turistas_corregido/dfmensualL.turistas_corregido.sum(),color="red")
    ax2["L"].plot(dfmensualL.Month,dfmensualL.PUD/dfmensualL.PUD.sum(),color="blue")
    ax2["L"].title.set_text(str(dfmensualL.SITE_NAME.unique()))
    # ax2["L"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["M"].plot(dfmensualM.Month,dfmensualM.Visitantes/dfmensualM.Visitantes.sum(),color="black")
    ax2["M"].plot(dfmensualM.Month,dfmensualM.turistas_corregido/dfmensualM.turistas_corregido.sum(),color="red")
    ax2["M"].plot(dfmensualM.Month,dfmensualM.PUD/dfmensualM.PUD.sum(),color="blue")
    ax2["M"].title.set_text(str(dfmensualM.SITE_NAME.unique()))
    # ax2["M"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["N"].plot(dfmensualN.Month,dfmensualN.Visitantes/dfmensualN.Visitantes.sum(),color="black")
    ax2["N"].plot(dfmensualN.Month,dfmensualN.turistas_corregido/dfmensualN.turistas_corregido.sum(),color="red")
    ax2["N"].plot(dfmensualN.Month,dfmensualN.PUD/dfmensualN.PUD.sum(),color="blue")
    ax2["N"].title.set_text(str(dfmensualN.SITE_NAME.unique()))
    # ax2["N"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["O"].plot(dfmensualO.Month,dfmensualO.Visitantes/dfmensualO.Visitantes.sum(),color="black")
    ax2["O"].plot(dfmensualO.Month,dfmensualO.turistas_corregido/dfmensualO.turistas_corregido.sum(),color="red")
    ax2["O"].plot(dfmensualO.Month,dfmensualO.PUD/dfmensualO.PUD.sum(),color="blue")
    ax2["O"].title.set_text(str(dfmensualO.SITE_NAME.unique()))
    # ax2["O"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["P"].plot(dfmensualP.Month,dfmensualP.Visitantes/dfmensualP.Visitantes.sum(),color="black")
    ax2["P"].plot(dfmensualP.Month,dfmensualP.turistas_corregido/dfmensualP.turistas_corregido.sum(),color="red")
    ax2["P"].plot(dfmensualP.Month,dfmensualP.PUD/dfmensualP.PUD.sum(),color="blue")
    ax2["P"].title.set_text(str(dfmensualP.SITE_NAME.unique()))
    # ax2["P"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    plt.show()
































