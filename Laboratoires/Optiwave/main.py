from Devoirs.Devoir1.Num1 import miyagi
import numpy as np


# Calculer n_eff
def get_ref_index(u, k0, a, n1):
    return np.sqrt((k0**2 * n1**2 - (u / a)**2)/(k0**2))


# Parametètres de laser
wl = 1.55e-06
k0 = 2 * np.pi / wl

n1_1 = 1.4519
n2 = 1.44402
a_1 = 2.35e-6
a_2 = 3.69e-6

NA = np.sqrt(n1_1 ** 2 - n2 ** 2)
V = k0 * NA * a_1
u = miyagi(V, True)
n_eff = get_ref_index(u, k0, a_1, n1_1)
print(f"L'indice effectif de la première fibre est {round(n_eff, 6)}")


best_n = 0
best_n_eff = 0
best_delta = 10000
for n1 in np.linspace(n2, 1.5, 10000):
    NA = np.sqrt(n1**2 - n2**2)
    V = k0 * NA * a_2
    mi_u = miyagi(V, True)
    new_n = get_ref_index(mi_u, k0, a_2, n1)

    if abs(new_n - n_eff) < best_delta:
        best_n = n1
        best_n_eff = new_n
        best_delta = abs(new_n - n_eff)


print(f"L'indice du coeur (n1) de la deuxième fibre est {best_n}")
print(f'La différence entre le n_eff des deux fibres serait donc de {best_n_eff - n_eff}')
