import matplotlib.pyplot as plt
from Laboratoires.utils import read_txt_data
from scipy.stats import linregress
from scipy.signal import argrelextrema, savgol_filter
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
theta_array = np.linspace(min(angle_pol) - 5, max(angle_pol) + 5, 1000)
ellipticity_theo = theo_ellipticity(theta_array, delta)
print(f'Différence de phase delta avec connecteur standard: {round(delta, 4)} rad ({delta / np.pi}π)')

plt.figure()
plt.errorbar(angle_pol, ellipticity, xerr=4, yerr=0.01, label='Courbe expérimentale',
             color='black', fmt='none', zorder=2, elinewidth=1)
plt.plot(theta_array, ellipticity_theo, label='Courbe théorique', zorder=1)
plt.ylabel('Ellipticité [-]')
plt.xlabel('Angle de la polarisation incidente [°]')
plt.xlim([min(theta_array), max(theta_array)])
plt.legend(loc='upper right')
plt.savefig('figs/3_1_pince.pdf', bbox_inches='tight')


# Sans pince
angle_pol_reduced_p, ellipticity_reduced_p = read_txt_data('data/3-1-nopince.txt')
angle_pol_reduced_p = (angle_pol_reduced_p - 140) * 2
delta_reduced_p = theo_delta(ellipticity_reduced_p)
theta_array = np.linspace(min(angle_pol_reduced_p) - 5, max(angle_pol_reduced_p) + 5, 1000)
ellipticity_theo_reduced_p = theo_ellipticity(theta_array, delta_reduced_p)
print(f'Différence de phase delta avec pression réduite: {round(delta_reduced_p, 4)} rad ({delta_reduced_p / np.pi}π)')

plt.figure()
plt.errorbar(angle_pol_reduced_p, ellipticity_reduced_p, xerr=4, yerr=0.01, label='Courbe expérimentale',
             color='black', fmt='none', zorder=2, elinewidth=1)
plt.plot(theta_array, ellipticity_theo_reduced_p, label='Courbe théorique', zorder=1)
plt.ylabel('Ellipticité [-]')
plt.xlabel('Angle de la polarisation incidente [°]')
plt.xlim([min(theta_array), max(theta_array)])
plt.legend(loc='upper right')
plt.savefig('figs/3_1_no_pince.pdf', bbox_inches='tight')
print('')


# Section 3.2 - Lames à retard

# Lame lambda / 4
R_l4 = 8 * K * np.pi * r_fibre_lb**2 / wl
print(f'Rayon de la boucle pour lambda/4: {round(R_l4 * 100, 4)} cm')

angle_pol_l4, ellipticity_l4 = read_txt_data('data/3-2-nopince.txt')
angle_pol_l4 = (angle_pol_l4 - 55) * 2
theta_array = np.linspace(min(angle_pol_l4) -5, max(angle_pol_l4) + 5, 1000)
delta_l4_theo = np.pi / 2       # On pose un déphasage théorique de pi / 2

plt.figure()
plt.errorbar(angle_pol_l4, ellipticity_l4, xerr=4, yerr=0.01, label='Courbe expérimentale',
             color='black', fmt='none', zorder=2, elinewidth=1)
plt.plot(theta_array, theo_ellipticity(theta_array, delta_l4_theo), label='Courbe théorique', zorder=1)
plt.ylabel('Ellipticité [-]')
plt.xlabel('Angle de la polarisation incidente [°]')
plt.xlim([min(theta_array), max(theta_array)])
plt.legend(loc='upper right')
plt.savefig('figs/3_2_no_pince.pdf', bbox_inches='tight')


# Lame lambda / 2
angle_pol_l2, ellipticity_l2 = read_txt_data('data/3-2-l2.txt')
angle_pol_l2 = (angle_pol_l2 - 90) * 2
theta_array = np.linspace(min(angle_pol_l2)-5, max(angle_pol_l2)+5, 1000)
delta_l2_theo = np.pi       # On pose un déphasage théorique de pi

plt.figure()
plt.errorbar(angle_pol_l2, ellipticity_l2, xerr=4, yerr=0.01, label='Courbe expérimentale',
             color='black', fmt='none', zorder=2, elinewidth=1)
plt.plot(theta_array, theo_ellipticity(theta_array, delta_l2_theo), label='Courbe théorique', zorder=1)
plt.ylabel('Ellipticité [-]')
plt.xlabel('Angle de la polarisation incidente [°]')
plt.xlim([min(theta_array), max(theta_array)])
plt.legend(loc='upper right')
plt.savefig('figs/3_2_lambda_2.pdf', bbox_inches='tight')


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
plt.errorbar(theta_in, theta_out, xerr=2, yerr=1, label='Courbe expérimentale',
             color='black', fmt='none', zorder=2, elinewidth=1)
plt.plot(theta_array, theta_out_theo, label='Fit linéaire', zorder=1)
plt.ylabel('Angle de polarisation à la sortie [°]')
plt.xlabel('Angle de la boucle [°]')
plt.xlim([3, 92])
plt.text(60, 50, f'y = {round(lin_fit[0], 3)}x + {round(lin_fit[1], 3)}\nR$^2$={round(lin_fit[2], 3)}')
plt.legend()
plt.savefig('figs/3_2_theta.pdf', bbox_inches='tight')


# Battements
def normalize(y, y_filter):
    y_filter -= min(y)
    y -= min(y)
    y_filter /= max(y)
    y /= max(y)
    return y, y_filter

data = np.genfromtxt("data/battements.csv", skip_header=True, delimiter=',')
data = np.array([[element[0], element[1]] for element in data if 4 < element[0] < 41])
x = data[:, 0]
y = data[:, 1]

y_filter = savgol_filter(y, 151, 3)
y, y_filter = normalize(y, y_filter)
maxima = argrelextrema(y_filter, np.greater)

ma = [7.472, 13.8766, 20.2184, 26.4032, 32.7136, 37.3915]

ma_i = []
for i in range(len(x)):
    if x[i] in ma:
        ma_i.append(i)

period = (ma[-1] - ma[0])/(len(ma) - 1)
birefringence = wl / (period * 1e-3)
print(f'La biréfringence est de {birefringence}')

freq = 1 / period
print(f'La fréquence des battements est de {freq} mm^-1')

plt.figure()
plt.plot(x, y, linewidth=1, label='Signal brut')
plt.plot(x, y_filter, label='Signal filtré')
plt.plot(ma, y_filter[ma_i], 'x', color='black', label='Maxima')
plt.text(30, 0.05, r'$L_B = $' + f'{round(period, 2)} mm')
plt.ylabel('Intensité normalisée [-]')
plt.xlabel('Distance z [mm]')
plt.xlim([x[0], x[-1]])
plt.legend()
plt.savefig('figs/battements.pdf', bbox_inches='tight')
plt.show()
