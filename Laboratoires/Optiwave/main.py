from Devoirs.Devoir1.Num1 import miyagi
import numpy as np


# Parametètres de laser
wl = 1.55e-06
k0 = 2 * np.pi / wl

n_eff = 1.4465
n2 = 1.44402
a = 3.69e-6


def get_ref_index(l, u, k0, a, n1):
    # Calculer n_eff
    return np.sqrt((k0**2 * n1**2 - (u / a)**2)/(k0**2))

best_n = 0
best_n_eff = 0
best_delta = 10000
for n1 in np.linspace(n2, 1.5, 10000):
    NA = np.sqrt(n1**2 - n2**2)
    V = k0 * NA * a
    mi_u = miyagi(V, True)
    new_n = get_ref_index(1, mi_u, k0, a, n1)

    if abs(new_n - n_eff) < best_delta:
        best_n = n1
        best_n_eff = new_n
        best_delta = abs(new_n - n_eff)


print('Best n1:', best_n)
print(best_n_eff)
