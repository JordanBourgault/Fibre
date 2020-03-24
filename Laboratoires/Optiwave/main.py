from Devoirs.Devoir1.Num1 import miyagi
import numpy as np
from scipy.special import kv


# Calculer n_eff
def get_ref_index(u, k0, a, n1):
    return np.sqrt((k0**2 * n1**2 - (u / a)**2)/(k0**2))


# Parametètres
wl = 1.55e-06
k0 = 2 * np.pi / wl

n1_1 = 1.4519
n2 = 1.44402
a_1 = 2.4e-6
a_2 = 3.69e-6
d = 12e-6

# Calcul n_eff de la première fibre (qui doit être égal au n_eff de la deuxième fibre)
NA = np.sqrt(n1_1 ** 2 - n2 ** 2)
V_1 = k0 * NA * a_1
u_1 = miyagi(V_1, True)
n_eff = get_ref_index(u_1, k0, a_1, n1_1)
print(f"L'indice effectif de la première fibre est {round(n_eff, 6)}")


best_n = 0
best_n_eff = 0
best_V = 0
best_u = 0
best_delta = 10000
for n1 in np.linspace(n2, 1.5, 10000):
    NA = np.sqrt(n1**2 - n2**2)
    V = k0 * NA * a_2
    mi_u = miyagi(V, True)
    new_n = get_ref_index(mi_u, k0, a_2, n1)

    if abs(new_n - n_eff) < best_delta:
        best_n = n1
        best_V = V
        best_u = mi_u
        best_n_eff = new_n
        best_delta = abs(new_n - n_eff)

n1_2 = best_n
V_2 = best_V
u_2 = best_u
print(f"L'indice du coeur (n1) de la deuxième fibre est {n1_2}")
print(f'La différence entre le n_eff des deux fibres serait donc de {best_n_eff - n_eff}')


def coupling(a1, a2, n1_1, n1_2, n2, V1, V2, u1, u2, d):
    delta_1 = (n1_1**2 - n2**2) / (2 * n1_1**2)
    delta_2 = (n1_2 ** 2 - n2 ** 2) / (2 * n1_2 ** 2)
    w1 = np.sqrt(V1**2 - u1**2)
    w2 = np.sqrt(V2**2 - u2**2)

    first = np.sqrt(2 / (a1 * a2))
    second = (delta_1 * delta_2 / (V1**6 * V2**6))**(1/4)
    third = u1 * u2 * kv(0, w1 * d / a1) / (kv(1, w1) * kv(1, w2))
    return first * second * third


C = coupling(a_1, a_2, n1_1, n1_2, n2, V_1, V_2, u_1, u_2, d)
print(f"Le coefficient de couplage est de {C}")
print(f"La longueur de couplage est de {1 / (2 * C) * 1e3} mm")
