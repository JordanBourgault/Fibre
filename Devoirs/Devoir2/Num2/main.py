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
    return np.real(num / denum)


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
    r = reflextivity(wl, n_eff)
    ref.append(r)
    new_wl = bragg_lambda(n_eff, Lambda, 1)
    if abs(new_wl - wl) < best_delta:
        best_delta = abs(new_wl - wl)
        best_wl = wl

print(f"La longueur d'onde de Bragg de premier ordre est de {best_wl * 1e6} µm")


# b)
plt.figure()
plt.plot(wavelengths * 1e6, ref)
plt.xlabel("Longueur d'onde [µm]")
plt.ylabel("Réfletivité [-]")
plt.xlim(([min(wavelengths) * 1e6, max(wavelengths) * 1e6]))
plt.savefig('figs/Num2_reflextivite.pdf', bbox_inches='tight')
print(f"La réflectivité maximale est de {max(ref)} à {round(wavelengths[np.argmax(ref)] * 1e6, 4)} µm")


plt.figure()
pertes = 10 * np.log10(1 - np.array(ref))
plt.plot(wavelengths * 1e6, pertes)
plt.ylabel("Pertes en transmission [dB]")
plt.xlabel("Longueur d'onde [µm]")
plt.xlim(([min(wavelengths) * 1e6, max(wavelengths) * 1e6]))
plt.savefig('figs/Num2_pertes.pdf', bbox_inches='tight')
print(f'La perte maximale est de {np.real(min(pertes))} dB à {round(wavelengths[np.argmax(pertes)] * 1e6, 4)} µm')


# c)
wavelengths = np.linspace(500e-9, 2000e-9, 10000)
best_wl_1 = 0
best_delta_1 = 10000000
best_wl_2 = 0
best_delta_2 = 10000000

for wl in wavelengths:
    n1 = sellmeier(SiO2, GeO2, x_germ, wl)
    n2 = sellmeier(SiO2, GeO2, 0, wl)
    NA = np.sqrt(n1 ** 2 - n2 ** 2)
    k0 = 2 * np.pi / wl
    V = k0 * NA * a
    u = miyagi(V)

    u_1 = u[0][0]
    n_eff_1 = get_ref_index(u_1, k0, n1, a)
    new_wl = bragg_lambda(n_eff_1, Lambda, 2)
    if abs(new_wl - wl) < best_delta_1:
        best_delta_1 = abs(new_wl - wl)
        best_wl_1 = wl

    try:
        u_2 = u[1][0]
        n_eff_2 = get_ref_index(u_2, k0, n1, a)
        new_wl = bragg_lambda(n_eff_2, Lambda, 2)
        if abs(new_wl - wl) < best_delta_2:
            best_delta_2 = abs(new_wl - wl)
            best_wl_2 = wl
    except IndexError:
        pass

print(f"La longueur d'onde de Bragg de deuxième ordre est de {best_wl_1 * 1e6} µm et {best_wl_2 * 1e6} µm")

plt.show()
