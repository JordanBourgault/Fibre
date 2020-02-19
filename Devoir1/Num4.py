import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv
from Devoir1.Num1 import get_u


# Params
SiO2 = {'A': [0.696166, 0.407942, 0.897479],
        'l': [0.068404e-6, 0.116241e-6, 9.896161e-6]}

GeO2 = {'A': [0.806866, 0.718158, 0.854168],
        'l': [0.068972e-6, 0.153966e-6, 11.84193e-6]}

c = 3e8
a = 3e-6


def sellmeier(mat1, mat2, x, wl):
    n2 = 0
    for i in range(3):
        n2 += (mat1['A'][i] + x * (mat2['A'][i] - mat1['A'][i])) * wl**2 \
              / (wl**2 - (mat1['l'][i] + x * (mat2['l'][i] - mat1['l'][i]))**2)

    n = np.sqrt(n2 + 1)
    return n


def material_dispersion(mat1, mat2, wl, derivative_int=0.01e-6):
    d1 = (sellmeier(mat1, mat2, 0, wl) - sellmeier(mat1, mat2, 0, wl-derivative_int)) / derivative_int
    d2 = (sellmeier(mat1, mat2, 0, wl+derivative_int) - sellmeier(mat1, mat2, 0, wl)) / derivative_int

    return ((-wl / c) * (d2 - d1) / derivative_int) * 1e6


def guiding_dispersion(mat1, mat2, x, wl):
    k0 = 2 * np.pi / wl
    n1 = sellmeier(mat1, mat2, x, wl)
    n2 = sellmeier(mat1, mat2, 0, wl)
    delta = (n1**2 - n2**2) / (2 * n1**2)
    V_arr = k0 * np.sqrt(n1**2 - n2**2) * a
    values = np.linspace(0, max(V_arr), 1000)

    u = np.array([get_u(values, np.sqrt(V**2 - values**2), 0, V)[0][0] for V in V_arr])
    w = np.sqrt(V_arr**2 - u**2)
    psi = (kv(0, w)) ** 2 / (kv(1, w) * kv(-1, w))

    dVb = 1 - (u/V_arr)**2 * (1 - 2*psi)
    VddVb = 2*(u/V_arr)**2 * (psi * (1 - 2*psi) + 2/w * (w**2 + u**2 * psi) * np.sqrt(psi) * (psi + 1/w * np.sqrt(psi) - 1))

    return delta * (material_dispersion(mat1, mat2, wl) * dVb - n2/(c * wl) * VddVb * 1e6)


if __name__ == '__main__':
    wl = np.linspace(1e-6, 1.5e-6, 1000)
    D_M = material_dispersion(SiO2, GeO2, wl, 0.0001e-6)
    fig = 1
    for concentration in [0.02, 0.08]:
        D_W = guiding_dispersion(SiO2, GeO2, concentration, wl)
        D = D_M + D_W

        plt.figure(fig)
        fig += 1
        plt.plot(wl*1e6, D, label='Dispersion totale')
        plt.plot(wl*1e6, D_M, label='Dispersion matérielle')
        plt.plot(wl*1e6, D_W, label='Dispersion de guidage')
        plt.title(f'{concentration*100}% molaire GeO$_2$')
        plt.ylabel('Dispersion [ps / km nm]')
        plt.xlabel("Longueur d'onde [µm]")
        plt.legend()
        plt.savefig(f'figs/num4_{concentration}.png')

    plt.show()
