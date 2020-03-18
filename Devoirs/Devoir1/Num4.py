import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv
from Devoirs.Devoir1.Num1 import get_u

# Params
SiO2 = {'A': [0.696166, 0.407942, 0.897479],
        'l': [0.068404e-6, 0.116241e-6, 9.896161e-6]}

GeO2 = {'A': [0.806866, 0.718158, 0.854168],
        'l': [0.068972e-6, 0.153966e-6, 11.84193e-6]}

c = 3e8
a = 3e-6


def sellmeier(mat1, mat2, x, wl):
    n = 0
    for i in range(3):
        n += (mat1['A'][i] + x * (mat2['A'][i] - mat1['A'][i])) * wl**2 \
              / (wl**2 - (mat1['l'][i] + x * (mat2['l'][i] - mat1['l'][i]))**2)
    return np.sqrt(n + 1)


def material_dispersion(mat1, mat2, wl, derivative_int=0.0001e-6):
    d1 = (sellmeier(mat1, mat2, 0, wl) - sellmeier(mat1, mat2, 0, wl-derivative_int)) / derivative_int
    d2 = (sellmeier(mat1, mat2, 0, wl+derivative_int) - sellmeier(mat1, mat2, 0, wl)) / derivative_int
    return ((-wl / c) * (d2 - d1) / derivative_int) * 1e6


def guiding_dispersion(mat1, mat2, delta, n2, u, w, V, wl):
    psi = (kv(0, w)) ** 2 / (kv(1, w) * kv(-1, w))
    dVb = 1 - (u/V)**2 * (1 - 2*psi)
    VddVb = 2*(u/V)**2 * (psi*(1 - 2*psi) + 2/w*(w**2 + u**2 * psi) * np.sqrt(psi) * (psi + 1/w * np.sqrt(psi) - 1))
    return delta * (material_dispersion(mat1, mat2, wl) * dVb - n2/(c * wl) * VddVb * 1e6)


def propagation_delay(L, ng2, delta, dVb, n2, P, b):
    return L/c * (ng2 * (1 + delta * dVb) - n2 * delta * P / 2 * (b + dVb))


def get_params(mat1, mat2, x, wl, L, derivative_int=0.0001e-6):
    k0 = 2 * np.pi / wl

    n1 = sellmeier(mat1, mat2, x, wl)
    n2 = sellmeier(mat1, mat2, 0, wl)
    n1_d = sellmeier(mat1, mat2, x, wl - derivative_int)
    n2_d = sellmeier(mat1, mat2, 0, wl - derivative_int)
    ng2 = n2 - wl * (n2 - n2_d) / derivative_int

    delta = (n1**2 - n2**2) / (2 * n1**2)
    ddelta = (delta - (n1_d ** 2 - n2_d ** 2) / (2 * n1_d ** 2)) / derivative_int
    P = wl / delta * ddelta

    V_arr = k0 * np.sqrt(n1**2 - n2**2) * a
    values = np.linspace(0, max(V_arr), 1000)

    u = np.array([get_u(values, np.sqrt(V**2 - values**2), 0, V)[0][0] for V in V_arr])
    w = np.sqrt(V_arr**2 - u**2)
    b = 1 - (u ** 2 / V_arr ** 2)
    psi = (kv(0, w)) ** 2 / (kv(1, w) * kv(-1, w))
    dVb = 1 - (u / V_arr) ** 2 * (1 - 2 * psi)
    return guiding_dispersion(mat1, mat2, delta, n2, u, w, V_arr, wl), propagation_delay(L, ng2, delta, dVb, n2, P, b)


if __name__ == '__main__':
    wl_range = np.array([1e-6, 1.5e-6])
    wl = np.linspace(wl_range[0], wl_range[1], 1000)
    D_M = material_dispersion(SiO2, GeO2, wl, 0.0001e-6)
    fig = 1
    for concentration in [0.02, 0.08]:
        D_W, T_g = get_params(SiO2, GeO2, concentration, wl, 10e3)
        D = D_M + D_W

        plt.figure(fig)
        fig += 1
        plt.plot(wl*1e6, D, label='Dispersion totale')
        plt.plot(wl*1e6, D_M, label='Dispersion matérielle')
        plt.plot(wl*1e6, D_W, label='Dispersion de guidage')
        plt.hlines(0, wl_range[0]*1e6, wl_range[1]*1e6, linestyles='dashed', linewidth=1)
        plt.title(f'{concentration*100}% molaire GeO$_2$')
        plt.ylabel('Dispersion [ps / km nm]')
        plt.xlabel("Longueur d'onde [µm]")
        plt.xlim(wl_range * 1e6)
        plt.legend()
        plt.savefig(f'figs/num4_{concentration}.pdf', bbox_inches='tight')

        plt.figure(0)
        plt.plot(wl*1e6, T_g*1e6, label=f'{concentration*100}% molaire de GeO$_2$')
        plt.title('Délai de propagation sur 10 km')
        plt.ylabel('Délai [µs]')
        plt.xlabel("Longueur d'onde [µm]")
        plt.xlim(wl_range * 1e6)
        plt.ylim([48.75, 49.3])
        plt.legend()
        plt.savefig('figs/num4_propagation.pdf', bbox_inches='tight')

    plt.show()
