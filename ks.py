import numpy as np
from scipy.stats import ks_2samp

# Frecuencias relativas de las dos fuentes
fuente_a = np.array([0.9])
fuente_b = np.array([0.01])

# Realizar el test KS
ks_statistic, p_value = ks_2samp(fuente_a, fuente_b,alternative="two-sided",method="exact")

print(f"Estadística KS: {ks_statistic}")
print(f"Valor p: {p_value}")

# Interpretación del resultado
if p_value < 0.05:
    print("Las distribuciones son significativamente diferentes.")
else:
    print("No hay evidencia suficiente para afirmar que las distribuciones son diferentes.")
