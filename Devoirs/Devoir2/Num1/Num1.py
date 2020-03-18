from Devoirs.Devoir1.Num4 import sellmeier, SiO2, GeO2, material_dispersion, guiding_dispersion
from Devoirs.Devoir1.Num1 import miyagi
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv


c = 3e8
a = 4.6e-6
a_clad = 62.5e-6
wl = 1.55e-6
T_0 = 1e-12
x_germ = 0.04

n1 = sellmeier(SiO2, GeO2, x_germ, wl)
n2 = sellmeier(SiO2, GeO2, 0, wl)
NA = np.sqrt(n1 ** 2 - n2 ** 2)
k0 = 2 * np.pi / wl
V = k0 * NA * a
u = miyagi(V, True)
w = np.sqrt(V**2 - u**2)
print(n1, n2, V)


def birefringence(n1, a, R):
    p11 = 0.12
    p12 = 0.27
    nu = 0.16
    return abs(n1**3 / 4 * (p11 - p12) * (1 + nu) * (a / R)**2)


def length(T_0, n1, a, R):
    return T_0 * c / birefringence(n1, a, R)


# a)

plt.figure()
radius = np.linspace(0, 10e-2, 1000)
len_r = length(2 * T_0, n1, a_clad, radius)
plt.plot(radius, len_r)
plt.xlabel('Rayon de courbure [m]')
plt.ylabel('Longueur de la fibre [m]')
plt.xlim([radius[0], radius[-1]])
plt.savefig(f'figs/rad_{T_0}.pdf', bbox_inches='tight')


# b)
def intramodal_dispersion(z):
    delta = (n1 ** 2 - n2 ** 2) / (2 * n1 ** 2)
    total_dispersion = material_dispersion(SiO2, GeO2, wl) + guiding_dispersion(SiO2, GeO2, delta, n2, u, w, V, wl)
    total_dispersion *= 1e-12 * 1e9
    beta_2 = -((wl)**2) * total_dispersion / (2 * np.pi * c)
    L_D = T_0**2 / abs(beta_2)
    print(L_D)
    return T_0 * np.sqrt(1 + (z/L_D)**2)


plt.figure()
z_arr = np.linspace(0, 0.1, 1000)
disp = intramodal_dispersion(z_arr)
plt.plot(z_arr * 1e3, disp * 1e12)
plt.xlabel('Longueur de la fibre [mm]')
plt.ylabel('Dispersion intramodale [ps]')
plt.xlim([z_arr[0] * 1e3, z_arr[-1] * 1e3])
plt.savefig(f'figs/disp_{T_0}.pdf', bbox_inches='tight')


# d)
def pertes_courbure(L):
    loss_arr = []
    for L_val in L:
        for i in range(len(radius)):
            if len_r[i] > L_val:
                R = radius[i]
                break

        A_c = 0.5 * np.sqrt(np.pi / (a * w**3)) * (u / (w * kv(1, w)))**2
        K = 4 * abs(n1 - n2) * w**3 / (3 * a * V**2 * n2)
        loss = A_c * np.exp(-K * R) / np.sqrt(R)
        loss_arr.append(loss * L_val / 1000)

    return loss_arr


def pertes_rayleigh(L):
    return ((0.75 + 66 * (n1 - n2)) * (wl * 1e6)**-4) * L


plt.figure()
radius = np.linspace(0, 1e-2, 1000)
len_r = length(2 * T_0, n1, a_clad, radius)
courb = np.array(pertes_courbure(len_r))
rayleigh = np.array(pertes_rayleigh(len_r))
plt.plot(len_r, courb, label='Pertes de courbure')
plt.plot(len_r, rayleigh, label='Pertes de Rayleigh')
plt.plot(len_r, courb + rayleigh, label='Pertes totales')
plt.xlabel('Longueur de la fibre [m]')
plt.ylabel('Pertes [dB]')
plt.legend()
plt.savefig(f'figs/pertes_{T_0}.pdf', bbox_inches='tight')
plt.show()
