import numpy as np
import matplotlib.pyplot as plt

from Devoirs.Devoir1.Num1 import miyagi, get_ref_index
from Devoirs.Devoir1.Num4 import sellmeier, SiO2, GeO2


phase_mask = 930e-9
Lambda = phase_mask / 2
L = 5e-3

delta_n = 5e-4
nu = 1

a = 5.0e-6 / 2
x_germ = 0.06


def bragg_lambda(n_eff, Lambda, m):
    return n_eff * 2 * Lambda / m


def reflextivity(wl, n_eff):
    dn_eff = delta_n / 2
    sigma = (2 * np.pi * (n_eff + dn_eff) / wl) - (np.pi / Lambda)
    kappa = np.pi * nu * dn_eff / wl
    num = np.square(np.sinh(np.lib.scimath.sqrt(kappa**2 - sigma**2) * L), casting='unsafe')
    denum = np.square(np.cosh(np.lib.scimath.sqrt(kappa**2 - sigma**2) * L)) - sigma**2 / kappa**2
    return num / denum


# a)
best_wl = 0
best_delta = 100000
ref = []
wavelengths = np.linspace(1347e-9, 1350e-9, 10000)
for wl in wavelengths:
    n1 = sellmeier(SiO2, GeO2, x_germ, wl)
    n2 = sellmeier(SiO2, GeO2, 0, wl)
    NA = np.sqrt(n1 ** 2 - n2 ** 2)
    k0 = 2 * np.pi / wl
    V = k0 * NA * a
    u = miyagi(V, True)
    n_eff = get_ref_index(u, k0, n1, a)
    ref.append(reflextivity(wl, n_eff))
    new_wl = bragg_lambda(n_eff, Lambda, 1)
    if abs(new_wl - wl) < best_delta:
        best_delta = abs(new_wl - wl)
        best_wl = wl

print(f"La longueur d'onde de Bragg est de {best_wl * 1e6} nm")

plt.plot(wavelengths * 1e6, ref)
plt.show()
