from Laboratoires.utils import read_txt_data
from Devoirs.Devoir1.Num4 import sellmeier, SiO2
from Devoirs.Devoir1.Num1 import miyagi

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelmax
from scipy.special import kv


# Definition de la fonction gaussienne
def gaussian(x, a, b, c):
    return a * np.exp((-(x - b)**2) / (2 * c**2))


# Extraction des donnees experimentales du mode fondamental
x, x_err, I, I_err = read_txt_data('data/fondamental.txt')

# Fit des donnees experimentaes avec le modele de fonction gaussienne
fit_params, cov = curve_fit(gaussian, x, I)
print(f'Les parametres de la gaussienne sont: {fit_params}')
x_fit = np.linspace(min(x), max(x), 1000)
I_fit = gaussian(x_fit, *fit_params)


# Recentrage et normalisation des donnees pour que la gaussienne soit centree et normalisee
x -= fit_params[1]
x_fit -= fit_params[1]
I /= fit_params[0]
I_err /= fit_params[0]
I_fit /= fit_params[0]

# Calcul de la largeur de la gaussienne
height = 1 / np.e**2
width = x_fit[(np.abs(I_fit - height)).argmin()]
print(f'Le rayon de la gaussienne est de {round(width, 4)} mm')

# Calcul de la divergence
L = 45
theta = np.arctan(width / L)
print(f'La divergence est de {round(theta, 4)} rad')

# Calcul du rayon du l'etranglement du faisceau
lambda_0 = 632.8e-9
n = 1
w_0 = lambda_0 / (np.tan(theta) * np.pi * n)
print(f"La taille de l'etranglement est de {round(w_0 * 1e6, 4)} um")

# Calcul du rayon de la fibre
lambda_c = 575e-9
V_c = 2.4048
V = V_c * lambda_c / lambda_0
print(f'V = {V}')

a = w_0 / (0.65 + 1.619 / (V**(3/2)) + 2.879 / V**6)
print(f'Le rayon de la fibre est de {a * 1e6} um')

# Creation de la figure
plt.figure()
plt.errorbar(x, I, xerr=x_err, yerr=I_err, label='Donnees exp.',
             color='black', fmt='none', elinewidth=1, capsize=1.5)
plt.plot(x_fit, I_fit, label='Fit gaussien')
plt.xlabel('Position relative [mm]')
plt.xlim([-max(x), max(x)])
plt.ylabel(r'Intensite normalisee [-]')
plt.legend()
plt.savefig('figs/mode_fondamental.pdf', bbox_inches='tight')


# Extraction des donnees experimentales du patron en couplage
x_coup, x_coup_err, I_coup, I_coup_err = read_txt_data('data/couplage.txt')

# Calcul de la position des maxima
local_maxima = np.array(x_coup)[argrelmax(np.array(I_coup))]
local_maxima = local_maxima[2:]
print(f'Les maxima sont situe a {local_maxima} mm')
period = []
for i in range(len(local_maxima) - 1):
    period.append(local_maxima[i+1] - local_maxima[i])
avg_period = sum(period) / len(period)
print(f"La periode moyenne du patron d'interference observe est {round(avg_period, 4)} mm")

# Calcul de la distance entre les coeurs
L = 35
d = lambda_0 * L / avg_period
print(f'La distance estimee entre les deux coeurs est de {round(d * 1e6, 4)} um')

# Creation de la figure
plt.figure()
plt.errorbar(x_coup, I_coup, xerr=x_coup_err, yerr=I_coup_err, label='Donnees exp.',
             color='black', fmt='none', elinewidth=1, capsize=1.5)
plt.xlabel('Position relative [mm]')
plt.xlim([min(x_coup), max(x_coup)])
plt.ylabel(r'Intensite [uW]')
plt.savefig('figs/couplage.pdf', bbox_inches='tight')


# Calcul du coefficient de couplage
NA = V * lambda_0 / (2 * np.pi * a)
n2 = sellmeier(SiO2, SiO2, 0, lambda_0)
n1 = np.sqrt(NA**2 + n2**2)
delta = (n1**2 - n2**2) / (2 * n1**2)
u = miyagi(V, True)
w = np.sqrt(V**2 - u**2)

C = np.sqrt(2 * delta) * u**2 * kv(0, w * d / a) / (a * V**3 * (kv(1, w))**2)
print(f'Le coefficient de couplage est de {round(C, 3)} m^-1')

# Calcul de la longueur de couplage
L_c = np.pi / (2 * C)
print(f'La longueur de couplage est de {round(L_c * 1e2, 3)} cm')


# Calcul du coefficient de coupage obtenu par Valle et Drolet
a_real = 1.9e-6
d_real = 9.9e-6
NA_real = V * lambda_0 / (2 * np.pi * a_real)
n1_real = np.sqrt(NA_real**2 + n2**2)
delta_real = (n1_real**2 - n2**2) / (2 * n1_real**2)
C_real = np.sqrt(2 * delta_real) * u**2 * kv(0, w * d_real / a_real) / (a_real * V**3 * (kv(1, w))**2)
print(f'Le coefficient de couplage obtenu par Valle et Drolet est de {round(C_real, 3)} m^-1')
L_c_real = np.pi / (2 * C_real)
print(f'La longueur de couplage obtenu par Valle et Drolet est de {round(L_c_real * 1e2, 3)} cm')

plt.show()
