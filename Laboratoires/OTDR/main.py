from Laboratoires.utils import read_txt_data
from Devoirs.Devoir1.Num4 import sellmeier, SiO2
import numpy as np
import matplotlib.pyplot as plt


def alpha_c(R, delta_n, wl, wl_c):

    A_c = 30 * (delta_n)**(1/4) / np.sqrt(wl) * (wl_c/wl)**(3/2)
    K = 0.705 * (delta_n)**(3/2) / wl * (2.748 - 0.996 * wl / wl_c)**3
    return A_c / np.sqrt(R) * np.exp(-K * R)


# Monomode
a_1 = 8.7e-6 / 2
NA_1 = 0.113
wl_c_1 = 2 * np.pi * a_1 / (2.405) * NA_1

wl = 1310e-9
n2 = sellmeier(SiO2, SiO2, 0, wl)
n1 = np.sqrt(NA_1**2 + n2**2)
delta_n = n1 - n2

theo_r = np.linspace(1.2e-2, 2.3e-2)
theo_alpha = alpha_c(theo_r, delta_n, wl, wl_c_1)

r, P_1310, P_1550 = read_txt_data('data.txt')
r = np.array(r) * 1e-2

plt.plot(theo_r, theo_alpha, label='Théorique')
plt.plot(r, P_1310, label='Expérimentale')
plt.legend()
plt.xlabel('Rayon de courbure [cm]')
plt.ylabel('Pertes par courbure [dB/m]')
plt.xlim([min(r), max(r)])
plt.savefig('perte_1310.pdf', bbox_inches='tight')

plt.show()
