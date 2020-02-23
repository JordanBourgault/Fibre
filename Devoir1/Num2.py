import matplotlib.pyplot as plt
from scipy.special import jv
from Devoir1.Num1 import miyagi
import numpy as np


wl = 633e-9
k0 = 2 * np.pi / wl
NA_values = [0.04, 0.06, 0.08, 0.10, 0.12]
V = np.linspace(1.8, 2.6, 100)


def waist(a, u, w):
    return np.sqrt(a**2 * 2/3 * (jv(0, u) / (u * jv(1, u)) + 1/2 + 1/w**2 - 1/u**2))


for NA in NA_values:
    a = V / (k0 * NA)
    u = np.array([miyagi(V_val, True) for V_val in V])
    w = np.sqrt(V**2 - u**2)
    w_0 = waist(a, u, w)

    plt.figure(0)
    plt.plot(V, w_0*1e6, label=f'NA={NA}')
    plt.ylabel('Diamètre w$_0$ [µm]')
    plt.xlabel('Fréquance normalisée V [-]')
    plt.xlim([1.8, 2.6])
    plt.ylim([1, 5.8])
    plt.legend(ncol=3)

plt.savefig('figs/Num2_w0.pdf', bbox_inches='tight')
plt.show()
