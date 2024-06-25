import numpy as np
from scipy.stats import mannwhitneyu

# Definir las frecuencias relativas observadas para ambas fuentes de datos
# Fuente A
observed_A = np.array([0.49, 0.49, 0.02])

# Fuente B
observed_B = np.array([0.01, 0.01, 0.98])

# Aplicar la prueba U de Mann-Whitney
statistic, p_value = mannwhitneyu(observed_A, observed_B)

print(f"Estadístico U de Mann-Whitney: {statistic}")
print(f"Valor p: {p_value}")

# Interpretar el valor p
alpha = 0.05
if p_value < alpha:
    print("Las distribuciones son significativamente diferentes al nivel de significancia del 5%.")
else:
    print("No hay evidencia suficiente para afirmar que las distribuciones son diferentes.")
