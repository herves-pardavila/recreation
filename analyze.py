import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__== "__main__":
    
    main_path="/media/david/EXTERNAL_USB/doctorado/"
    #anual visitors
    df=pd.read_csv(main_path+"/recreation/recreation_ready.csv")
    print(df.Month.unique())
    dfanual=df.groupby(by=["IdOAPN"],as_index=False).sum(numeric_only=True)
    print(dfanual.Month.unique())
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.loglog(dfanual.Visitantes,dfanual.PUD,"*",label="Flickr PUD")
    ax.loglog(dfanual.Visitantes,dfanual.turistas_total,"*",label="INE data")
    ax.loglog(dfanual.Visitantes,dfanual.turistas_corregido,"*",label="INE data corregido")
    ax.loglog(dfanual.Visitantes,dfanual.IUD,"*",label="Instagram")
    ax.loglog(np.arange(1e5,8e7,10),np.arange(1e5,8e7,10),label="1-1 line")
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.PUD,dfanual.IdOAPN)]
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.turistas_total,dfanual.IdOAPN)]
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.turistas_corregido,dfanual.IdOAPN)]
    [plt.text(i,j,f"{k}") for (i,j,k) in zip(dfanual.Visitantes,dfanual.IUD,dfanual.IdOAPN)]
    ax.set_xlabel("Visitors")
    ax.set_ylabel("Estimated User-generated-data visitors")
    fig.legend()
    plt.show()

    #Monthly visitation per park
    dfmensual=df.groupby(by=["IdOAPN","Month"],as_index=False).sum(numeric_only=True)
    print(dfmensual.Month.unique())
    print(dfmensual)
    dfmensualA=dfmensual[dfmensual.IdOAPN=="Aigüestortes i Estany de Sant Maurici"]
    dfmensualB=dfmensual[dfmensual.IdOAPN=="Archipiélago de Cabrera"]
    dfmensualC=dfmensual[dfmensual.IdOAPN=="Cabañeros"]
    dfmensualD=dfmensual[dfmensual.IdOAPN=="Caldera de Taburiente"]
    dfmensualE=dfmensual[dfmensual.IdOAPN=="Doñana"]
    dfmensualF=dfmensual[dfmensual.IdOAPN=="Garajonay"]
    dfmensualG=dfmensual[dfmensual.IdOAPN=="Islas Atlánticas de Galicia"]
    dfmensualH=dfmensual[dfmensual.IdOAPN=="Monfragüe"]
    dfmensualI=dfmensual[dfmensual.IdOAPN=="Ordesa y Monte Perdido"]
    dfmensualJ=dfmensual[dfmensual.IdOAPN=="Picos de Europa"]
    dfmensualK=dfmensual[dfmensual.IdOAPN=="Sierra Nevada"]
    print(dfmensualK)
    dfmensualL=dfmensual[dfmensual.IdOAPN=="Sierra de Guadarrama"]
    dfmensualM=dfmensual[dfmensual.IdOAPN=="Sierra de las Nieves"]
    dfmensualN=dfmensual[dfmensual.IdOAPN=="Tablas de Daimiel"]
    dfmensualO=dfmensual[dfmensual.IdOAPN=="Teide National Park"]
    dfmensualP=dfmensual[dfmensual.IdOAPN=="Timanfaya"]

    fig2=plt.figure()
    ax2=fig2.subplot_mosaic("""ABCD
                         EFGH
                         IJKL
                         MNOP""")

    fig2.subplots_adjust(hspace=0.45,wspace=0.3,top=0.9,bottom=0.1,left=0.05,right=0.95)
    ax2["A"].tick_params(axis='both',labelsize=15)
    ax2["A"].plot(dfmensualA.Month,dfmensualA.Visitantes/dfmensualA.Visitantes.sum(),color="black")
    ax2["A"].plot(dfmensualA.Month,dfmensualA.turistas_corregido/dfmensualA.turistas_corregido.sum(),color="red")
    ax2["A"].plot(dfmensualA.Month,dfmensualA.PUD/dfmensualA.PUD.sum(),color="blue")
    ax2["A"].set_title(str(dfmensualA.IdOAPN.iloc[0]),fontsize=15)
    #ax2["A"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2["B"].tick_params(axis='both',labelsize=15)
    ax2["B"].plot(dfmensualB.Month,dfmensualB.Visitantes/dfmensualB.Visitantes.sum(),color="black")
    ax2["B"].plot(dfmensualB.Month,dfmensualB.turistas_corregido/dfmensualB.turistas_corregido.sum(),color="red")
    ax2["B"].plot(dfmensualB.Month,dfmensualB.PUD/dfmensualB.PUD.sum(),color="blue")
    ax2["B"].set_title(str(dfmensualB.IdOAPN.iloc[0]),fontsize=15)
    # ax2["B"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["C"].tick_params(axis='both',labelsize=15)
    ax2["C"].plot(dfmensualC.Month,dfmensualC.Visitantes/dfmensualC.Visitantes.sum(),color="black")
    ax2["C"].plot(dfmensualC.Month,dfmensualC.turistas_corregido/dfmensualC.turistas_corregido.sum(),color="red")
    ax2["C"].plot(dfmensualC.Month,dfmensualC.PUD/dfmensualC.PUD.sum(),color="blue")
    ax2["C"].set_title(str(dfmensualC.IdOAPN.iloc[0]),fontsize=15)
    # ax2["C"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    
    ax2["D"].tick_params(axis='both',labelsize=15)
    ax2["D"].plot(dfmensualD.Month,dfmensualD.Visitantes/dfmensualD.Visitantes.sum(),color="black")
    ax2["D"].plot(dfmensualD.Month,dfmensualD.turistas_corregido/dfmensualD.turistas_corregido.sum(),color="red")
    ax2["D"].plot(dfmensualD.Month,dfmensualD.PUD/dfmensualD.PUD.sum(),color="blue")
    ax2["D"].set_title(str(dfmensualD.IdOAPN.iloc[0]),fontsize=15)
    # ax2["D"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    
    ax2["E"].tick_params(axis='both',labelsize=15)
    ax2["E"].plot(dfmensualE.Month,dfmensualE.Visitantes/dfmensualE.Visitantes.sum(),color="black")
    ax2["E"].plot(dfmensualE.Month,dfmensualE.turistas_corregido/dfmensualE.turistas_corregido.sum(),color="red")
    ax2["E"].plot(dfmensualE.Month,dfmensualE.PUD/dfmensualE.PUD.sum(),color="blue")
    ax2["E"].set_title(str(dfmensualE.IdOAPN.iloc[0]),fontsize=15)
    # ax2["E"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    
    ax2["F"].tick_params(axis='both',labelsize=15)
    ax2["F"].plot(dfmensualF.Month,dfmensualF.Visitantes/dfmensualF.Visitantes.sum(),color="black")
    ax2["F"].plot(dfmensualF.Month,dfmensualF.turistas_corregido/dfmensualF.turistas_corregido.sum(),color="red")
    ax2["F"].plot(dfmensualF.Month,dfmensualF.PUD/dfmensualF.PUD.sum(),color="blue")
    ax2["F"].set_title(str(dfmensualF.IdOAPN.iloc[0]),fontsize=15)
    # ax2["F"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["G"].tick_params(axis='both',labelsize=15)
    ax2["G"].plot(dfmensualG.Month,dfmensualG.Visitantes/dfmensualG.Visitantes.sum(),color="black")
    ax2["G"].plot(dfmensualG.Month,dfmensualG.turistas_corregido/dfmensualG.turistas_corregido.sum(),color="red")
    ax2["G"].plot(dfmensualG.Month,dfmensualG.PUD/dfmensualG.PUD.sum(),color="blue")
    ax2["G"].plot(dfmensualG.Month,dfmensualG.IUD/dfmensualG.IUD.sum(),color="purple")
    ax2["G"].set_title(str(dfmensualG.IdOAPN.iloc[0]),fontsize=15)
    # ax2["G"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["H"].tick_params(axis='both',labelsize=15)
    ax2["H"].plot(dfmensualH.Month,dfmensualH.Visitantes/dfmensualH.Visitantes.sum(),color="black")
    ax2["H"].plot(dfmensualH.Month,dfmensualH.turistas_corregido/dfmensualH.turistas_corregido.sum(),color="red")
    ax2["H"].plot(dfmensualH.Month,dfmensualH.PUD/dfmensualH.PUD.sum(),color="blue")
    ax2["H"].set_title(str(dfmensualH.IdOAPN.iloc[0]),fontsize=15)
    # ax2["H"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["I"].tick_params(axis='both',labelsize=15)
    ax2["I"].plot(dfmensualI.Month,dfmensualI.Visitantes/dfmensualI.Visitantes.sum(),color="black")
    ax2["I"].plot(dfmensualI.Month,dfmensualI.turistas_corregido/dfmensualI.turistas_corregido.sum(),color="red")
    ax2["I"].plot(dfmensualI.Month,dfmensualI.PUD/dfmensualI.PUD.sum(),color="blue")
    ax2["I"].set_title(str(dfmensualI.IdOAPN.iloc[0]),fontsize=15)
    # ax2["I"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["J"].tick_params(axis='both',labelsize=15)
    ax2["J"].plot(dfmensualJ.Month,dfmensualJ.Visitantes/dfmensualJ.Visitantes.sum(),color="black")
    ax2["J"].plot(dfmensualJ.Month,dfmensualJ.turistas_corregido/dfmensualJ.turistas_corregido.sum(),color="red")
    ax2["J"].plot(dfmensualJ.Month,dfmensualJ.PUD/dfmensualJ.PUD.sum(),color="blue")
    ax2["J"].set_title(str(dfmensualJ.IdOAPN.iloc[0]),fontsize=15)
    # ax2["J"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["K"].tick_params(axis='both',labelsize=15)
    ax2["K"].plot(dfmensualK.Month,dfmensualK.Visitantes/dfmensualK.Visitantes.sum(),color="black")
    ax2["K"].plot(dfmensualK.Month,dfmensualK.turistas_corregido/dfmensualK.turistas_corregido.sum(),color="red")
    ax2["K"].plot(dfmensualK.Month,dfmensualK.PUD/dfmensualK.PUD.sum(),color="blue")
    ax2["K"].set_title(str(dfmensualK.IdOAPN.iloc[0]),fontsize=15)
    # ax2["K"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["L"].tick_params(axis='both',labelsize=15)
    ax2["L"].plot(dfmensualL.Month,dfmensualL.Visitantes/dfmensualL.Visitantes.sum(),color="black")
    ax2["L"].plot(dfmensualL.Month,dfmensualL.turistas_corregido/dfmensualL.turistas_corregido.sum(),color="red")
    ax2["L"].plot(dfmensualL.Month,dfmensualL.PUD/dfmensualL.PUD.sum(),color="blue")
    ax2["L"].set_title(str(dfmensualL.IdOAPN.iloc[0]),fontsize=15)
    # ax2["L"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])

    ax2["M"].tick_params(axis='both',labelsize=15)
    ax2["M"].plot(dfmensualM.Month,dfmensualM.Visitantes/dfmensualM.Visitantes.sum(),color="black")
    ax2["M"].plot(dfmensualM.Month,dfmensualM.turistas_corregido/dfmensualM.turistas_corregido.sum(),color="red")
    ax2["M"].plot(dfmensualM.Month,dfmensualM.PUD/dfmensualM.PUD.sum(),color="blue")
    ax2["M"].set_title(str(dfmensualM.IdOAPN.iloc[0]),fontsize=15)
    # ax2["M"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2["N"].tick_params(axis='both',labelsize=15)
    ax2["N"].plot(dfmensualN.Month,dfmensualN.Visitantes/dfmensualN.Visitantes.sum(),color="black")
    ax2["N"].plot(dfmensualN.Month,dfmensualN.turistas_corregido/dfmensualN.turistas_corregido.sum(),color="red")
    ax2["N"].plot(dfmensualN.Month,dfmensualN.PUD/dfmensualN.PUD.sum(),color="blue")
    ax2["N"].plot(dfmensualN.Month,dfmensualN.IUD/dfmensualN.IUD.sum(),color="purple")
    ax2["N"].set_title(str(dfmensualN.IdOAPN.iloc[0]),fontsize=15)
    # ax2["N"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2["O"].tick_params(axis='both',labelsize=15)
    ax2["O"].plot(dfmensualO.Month,dfmensualO.Visitantes/dfmensualO.Visitantes.sum(),color="black")
    ax2["O"].plot(dfmensualO.Month,dfmensualO.turistas_corregido/dfmensualO.turistas_corregido.sum(),color="red")
    ax2["O"].plot(dfmensualO.Month,dfmensualO.PUD/dfmensualO.PUD.sum(),color="blue")
    ax2["O"].set_title(str(dfmensualO.IdOAPN.iloc[0]),fontsize=15)
    # ax2["O"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2["P"].tick_params(axis='both',labelsize=15)
    ax2["P"].plot(dfmensualP.Month,dfmensualP.Visitantes/dfmensualP.Visitantes.sum(),color="black")
    ax2["P"].plot(dfmensualP.Month,dfmensualP.turistas_corregido/dfmensualP.turistas_corregido.sum(),color="red")
    ax2["P"].plot(dfmensualP.Month,dfmensualP.PUD/dfmensualP.PUD.sum(),color="blue")
    ax2["P"].plot(dfmensualP.Month,dfmensualP.IUD/dfmensualP.IUD.sum(),color="purple")
    ax2["P"].set_title(str(dfmensualP.IdOAPN.iloc[0]),fontsize=15)
    # ax2["P"].set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    #fig2.savefig("/home/usuario/Documentos/recreation/imagenes_paper.pdf")
    plt.show()
































