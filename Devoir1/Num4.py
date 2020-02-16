import numpy as np
import matplotlib.pyplot as plt


# Params
SiO2 = {'A': [0.696166, 0.407942, 0.897479],
        'l': [0.068404e-6, 0.116241e-6, 9.896161e-6]}

GeO2 = {'A': [0.806866, 0.718158, 0.854168],
        'l': [0.068972e-6, 0.153966e-6, 11.84193e-6]}

c = 3e8

def sellmeier(mat1, mat2, x, wl):
    n2 = 0
    for i in range(3):
        n2 += (mat1['A'][i] + x * (mat2['A'][i] - mat1['A'][i])) * wl**2 \
              / (wl**2 - (mat1['l'][i] + x * (mat2['l'][i] - mat1['l'][i]))**2)

    n = np.sqrt(n2 + 1)
    print(n)
    return n


def material_dispersion(mat1, mat2, wl, derivative_int):
    d1 = (sellmeier(mat1, mat2, 0, wl) - sellmeier(mat1, mat2, 0, wl-derivative_int)) / derivative_int
    d2 = (sellmeier(mat1, mat2, 0, wl+derivative_int) - sellmeier(mat1, mat2, 0, wl)) / derivative_int

    return ((-wl / c) * (d2 - d1) / derivative_int) * 1e6


wl = np.linspace(1e-6, 1.5e-6, 100)
D_M = material_dispersion(SiO2, GeO2, wl, 0.0001e-6)

plt.plot(wl*1e6, D_M)
plt.ylabel('Dispersion [ps / km nm]')
plt.xlabel("Longueur d'onde [µm]")
plt.show()
