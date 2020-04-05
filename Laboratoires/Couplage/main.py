from Laboratoires.utils import read_txt_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelmax


# Définition de la fonction gaussienne
def gaussian(x, a, b, c):
    return a * np.exp((-(x - b)**2) / (2 * c**2))


# Extraction des données expérimentales du mode fondamental
x, x_err, I, I_err = read_txt_data('data/fondamental.txt')

# Fit des données expérimentaes avec le modèle de fonction gaussienne
fit_params, cov = curve_fit(gaussian, x, I)
x_fit = np.linspace(min(x), max(x), 1000)
I_fit = gaussian(x_fit, *fit_params)

# Recentrage des données pour que la gaussienne soit centrée
x -= fit_params[1]
x_fit -= fit_params[1]

# Calcul de la largeur de la gaussienne
height = 1 / np.e**2
width = x_fit[(np.abs(I_fit - height)).argmin()]
print(f'Le rayon de la gaussienne est de {round(width, 4)} mm')

# Calcul de la divergence
L = 45
theta = np.arctan(width / L)
print(f'La divergence est de {round(theta, 4)} rad')

# Calcul du rayon du coeur de la fibre
lambda_0 = 632.8e-9
n = 1
r = lambda_0 / (np.tan(theta) * np.pi * n)
print(f'Le diamètre du coeur est de {round(r * 1e6 * 2, 4)} µm')

# Création de la figure
plt.figure()
plt.errorbar(x, I, xerr=x_err, yerr=I_err, label='Données exp.', color='black', fmt='none', elinewidth=1, capsize=1.5)
plt.plot(x_fit, I_fit, label='Fit gaussien')
plt.xlabel('Position relative [mm]')
plt.xlim([-max(x), max(x)])
plt.ylabel('Intensité [µW]')
plt.legend()
plt.savefig('figs/mode_fondamental.pdf')


# Extraction des données expérimentales du patron en couplage
x_coup, x_coup_err, I_coup, I_coup_err = read_txt_data('data/couplage.txt')

# Calcul de la position des maxima
local_maxima = np.array(x_coup)[argrelmax(np.array(I_coup))]
local_maxima = local_maxima[2:]
print(f'Les maxima sont situé à {local_maxima} mm')
period = []
for i in range(len(local_maxima) - 1):
    period.append(local_maxima[i+1] - local_maxima[i])
avg_period = sum(period) / len(period)
print(f"La période moyenne du patron d'interférence observé est {round(avg_period, 4)} mm")

# Calcul de la distance entre les coeurs
L = 35
d = lambda_0 * L / avg_period
print(f'La distance estimée entre les deux coeurs est de {round(d * 1e6, 4)} µm')

# Création de la figure
plt.figure()
plt.errorbar(x_coup, I_coup, xerr=x_coup_err, yerr=I_coup_err, label='Données exp.',
             color='black', fmt='none', elinewidth=1, capsize=1.5)
plt.xlabel('Position relative [mm]')
plt.xlim([min(x_coup), max(x_coup)])
plt.ylabel('Intensité [µW]')
plt.savefig('figs/couplage.pdf')

plt.show()
