import matplotlib.pyplot as plt
from Laboratoires.utils import read_txt_data
from scipy.stats import linregress
import numpy as np


# Paramètres généraux
wl = 632.8e-9

# Section 3 - Fibre faible biréfringence
K = 0.133
r_fibre_lb = 125e-6 / 2
NA_fibre_lb = 0.1


def theo_ellipticity(theta, delta):
    theta = np.deg2rad(theta)
    psi = np.arcsin(np.sin(2 * theta) * np.sin(delta)) / 2
    return np.tan(psi)

def theo_delta(ellipticity):
    return 2 * np.arctan(np.max(np.abs(ellipticity)))


# Section 3.1 - Ellipticité en fonction de l'angle de la lame demi-onde

# Avec pince
angle_pol, ellipticity = read_txt_data('data/3-1-pince.txt')
angle_pol = (angle_pol - 160) * 2       # Conversion de l'angle de la lame demi-onde en angle de polarisation incidente
delta = theo_delta(ellipticity)
theta_array = np.linspace(min(angle_pol), max(angle_pol), 1000)
ellipticity_theo = theo_ellipticity(theta_array, delta)
print(f'Différence de phase delta avec connecteur standard: {round(delta, 4)} rad ({delta / np.pi}π)')

plt.figure()
plt.plot(angle_pol, ellipticity, '.', label='Courbe expérimentale')
plt.plot(theta_array, ellipticity_theo, label='Courbe théorique')
plt.ylabel('Ellipticité [-]')
plt.xlabel('Angle de la polarisation incidente [°]')
plt.xlim([min(theta_array), max(theta_array)])
plt.legend(loc='upper right')


# Sans pince
angle_pol_reduced_p, ellipticity_reduced_p = read_txt_data('data/3-1-nopince.txt')
angle_pol_reduced_p = (angle_pol_reduced_p - 140) * 2
delta_reduced_p = theo_delta(ellipticity_reduced_p)
theta_array = np.linspace(min(angle_pol_reduced_p), max(angle_pol_reduced_p), 1000)
ellipticity_theo_reduced_p = theo_ellipticity(theta_array, delta_reduced_p)
print(f'Différence de phase delta avec pression réduite: {round(delta_reduced_p, 4)} rad ({delta_reduced_p / np.pi}π)')

plt.figure()
plt.plot(angle_pol_reduced_p, ellipticity_reduced_p, '.', label='Courbe expérimentale')
plt.plot(theta_array, ellipticity_theo_reduced_p, label='Courbe théorique')
plt.ylabel('Ellipticité [-]')
plt.xlabel('Angle de la polarisation incidente [°]')
plt.xlim([min(theta_array), max(theta_array)])
plt.legend(loc='upper right')
print('')


# Section 3.2 - Lames à retard

# Lame lambda / 4
R_l4 = 8 * K * np.pi * r_fibre_lb**2 / wl
print(f'Rayon de la boucle pour lambda/4: {round(R_l4 * 100, 4)} cm')

angle_pol_l4, ellipticity_l4 = read_txt_data('data/3-2-nopince.txt')
angle_pol_l4 = (angle_pol_l4 - 55) * 2
theta_array = np.linspace(min(angle_pol_l4), max(angle_pol_l4), 1000)
delta_l4_theo = np.pi / 2       # On pose un déphasage théorique de pi / 2

plt.figure()
plt.plot(angle_pol_l4, ellipticity_l4, '.', label='Courbe expérimentale')
plt.plot(theta_array, theo_ellipticity(theta_array, delta_l4_theo), label='Courbe théorique')
plt.ylabel('Ellipticité [-]')
plt.xlabel('Angle de la polarisation incidente [°]')
plt.xlim([min(theta_array), max(theta_array)])
plt.legend(loc='upper right')


# Lame lambda / 2
angle_pol_l2, ellipticity_l2 = read_txt_data('data/3-2-l2.txt')
angle_pol_l2 = (angle_pol_l2 - 90) * 2
theta_array = np.linspace(min(angle_pol_l2), max(angle_pol_l2), 1000)
delta_l2_theo = np.pi       # On pose un déphasage théorique de pi

plt.figure()
plt.plot(angle_pol_l2, ellipticity_l2, '.', label='Courbe expérimentale')
plt.plot(theta_array, theo_ellipticity(theta_array, delta_l2_theo), label='Courbe théorique')
plt.ylabel('Ellipticité [-]')
plt.xlabel('Angle de la polarisation incidente [°]')
plt.xlim([min(theta_array), max(theta_array)])
plt.legend(loc='upper right')


# Theta_out vs theta_in
theta_in, theta_out = read_txt_data('data/3-2-theta.txt')
for i in range(len(theta_in)):
    if theta_in[i] > 40:
        theta_out[i] += 180

lin_fit = linregress(theta_in, theta_out)
print(lin_fit)
theta_array = np.linspace(min(theta_in), max(theta_in))
theta_out_theo = lin_fit[0] * theta_array + lin_fit[1]


plt.figure()
plt.plot(theta_in, theta_out, '.', label='Courbe expérimentale')
plt.plot(theta_array, theta_out_theo, label='Fit linéaire')
plt.ylabel('Angle de polarisation à la sortie [°]')
plt.xlabel('Angle de la boucle [°]')
plt.text(60, 50, f'y = {round(lin_fit[0], 3)}x + {round(lin_fit[1], 3)}\nR$^2$={round(lin_fit[2], 3)}')
plt.legend()


plt.show()
