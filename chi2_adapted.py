import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

def chi_sq(o,e):
    return np.sum((o-e)*(o-e)/e)

df=pd.read_csv("/home/usuario/Documentos/recreation/recreation_ready.csv")
df=df[df.IdOAPN=="Monfragüe"]
df.Date=pd.to_datetime(df.Date)

df=df.groupby(by=["Year","IdOAPN"],as_index=False).sum(numeric_only=True)
print(df[["IdOAPN","Year","Visitantes"]])

# Definir las frecuencias relativas observadas y esperadas
observed = 19606*np.array([0.075,0.79,0.134])
print(observed)
expected = 19606*np.array([0.134,0.832,0.05])
print(expected)
chi_squared_observed=chi_sq(observed,expected)
print("El chi cuadrado es igual a ",chi_squared_observed)





# Determinar los grados de libertad
degrees_of_freedom = len(observed) - 1

# Obtener valores críticos de la distribución chi cuadrado para el nivel de significancia deseado
alpha_5 = 0.05
alpha_1 = 0.01
critical_value_5 = chi2.ppf(1 - alpha_5, degrees_of_freedom)
critical_value_1 = chi2.ppf(1 - alpha_1, degrees_of_freedom)

print(f"Valor crítico al 5% de significancia: {critical_value_5}")
print(f"Valor crítico al 1% de significancia: {critical_value_1}")

# Comparar con los valores críticos
if chi_squared_observed > critical_value_1:
    print("Las distribuciones son significativamente diferentes al nivel de significancia del 1%.")
elif chi_squared_observed > critical_value_5:
    print("Las distribuciones son significativamente diferentes al nivel de significancia del 5%.")
else:
    print("No hay evidencia suficiente para afirmar que las distribuciones son diferentes.")












# Número de simulaciones
num_simulations = 100000
Ns=[1000]
# Lista para almacenar los valores de chi cuadrado simulados

# for N in Ns:
#     chi_squared_values = []
#     for _ in range(num_simulations):
#         # Generar muestra sintética basada en las frecuencias esperadas
#         synthetic_relative = np.random.dirichlet(expected*N)
        
#         # Calcular el estadístico chi cuadrado relativo
#         chi_squared = np.sum((synthetic_relative - expected)*(synthetic_relative - expected)/expected)
      
#         chi_squared_values.append(chi_squared)

#     # Convertir a array numpy
#     chi_squared_values = np.array(chi_squared_values)
#     # Calcular percentiles para determinar valores críticos
#     critical_values = np.percentile(chi_squared_values, [95, 99])
#     print("Valores críticos para niveles de significancia 0.05 y 0.01:", critical_values)
    



# #visual
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.hist(chi_squared_values,bins=100000,density=True,stacked=True,histtype="step")

# plt.show()


