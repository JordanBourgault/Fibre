import matplotlib.pyplot as plt
from Laboratoires.utils import read_txt_data
import numpy as np


# Section 3.1 - Ellipticité en fonction de l'angle de la lame demi-onde

def theo_ellipticity(theta, delta):
    theta = np.deg2rad(theta)
    psi = np.arcsin(np.sin(2 * theta) * np.sin(delta)) / 2
    return np.tan(psi)


# Avec pince
angle_pol, ellipticity = read_txt_data('data/3-1-pince.txt')
angle_pol = (angle_pol - 160) * 2       # Conversion de l'angle de la lame demi-onde en angle de polarisation incidente
delta = 2 * np.arctan(np.max(np.abs(ellipticity)))
theta_array = np.linspace(min(angle_pol), max(angle_pol), 1000)
ellipticity_theo = theo_ellipticity(theta_array, delta)
print(f'Différence de phase delta avec connecteur standard: {round(delta, 4)} rad')

plt.figure(0)
plt.plot(angle_pol, ellipticity, label='Courbe expérimentale')
plt.plot(theta_array, ellipticity_theo, label='Courbe théorique')
plt.ylabel('Ellipticité [-]')
plt.xlabel('Angle de la polarisation incidente [°]')
plt.xlim([min(theta_array), max(theta_array)])
plt.legend(loc='upper right')


# Sans pince
angle_pol_reduced_p, ellipticity_reduced_p = read_txt_data('data/3-1-nopince.txt')
angle_pol_reduced_p = (angle_pol_reduced_p - 140) * 2
delta_reduced_p = 2 * np.arctan(np.max(np.abs(ellipticity_reduced_p)))
theta_array = np.linspace(min(angle_pol_reduced_p), max(angle_pol_reduced_p), 1000)
ellipticity_theo_reduced_p = theo_ellipticity(theta_array, delta_reduced_p)
print(f'Différence de phase delta avec pression réduite: {round(delta_reduced_p, 4)} rad')

plt.figure(1)
plt.plot(angle_pol_reduced_p, ellipticity_reduced_p, label='Courbe expérimentale')
plt.plot(theta_array, ellipticity_theo_reduced_p, label='Courbe théorique')
plt.ylabel('Ellipticité [-]')
plt.xlabel('Angle de la polarisation incidente [°]')
plt.xlim([min(theta_array), max(theta_array)])
plt.legend(loc='upper right')



# Section 3.2 - Lames à retard




plt.show()
