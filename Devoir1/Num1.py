from scipy.special import jv, kv
import numpy as np
import matplotlib.pyplot as plt

# Parametères de fibre
NA = 0.14
n2 = 1.4630
n1 = np.sqrt(NA**2 + n2**2)
a = 4.1e-6

# Parametètres de laser
wl = 488e-9
k0 = 2 * np.pi / wl

# Parametètres de propagation
V = k0 * NA * a
u = np.linspace(0, round(V + 3), 100000)
w = np.sqrt(V**2 - u**2)


# Trouve les valeurs de u correspondants aux points d'index
def get_intersects(u, idx, V):
    intersects = []
    intersect_list = u[idx][::2]            # Exclure les intersections d'asymptotes
    for u_val in intersect_list:
        if u_val > V:                       # Arrêter de chercher des points d'intersection après V
            break
        if u_val > 0:                       # Ne pas considérer l'intersection à 0
            intersects.append(u_val)
    return intersects


# Calcule les valeurs de u des différents modes qui se propagent
def get_u(u, w, l, V):
    with np.errstate(divide='ignore', invalid='ignore'):
        # Retourne le membre de gauche et le membre de droite de l'équation des modes LP
        left = jv(l, u) / (u * jv(l - 1, u))
        right = -kv(l, w) / (w * kv(l - 1, w))
        # Trouve les indices d'intersection entre le membre de gauche et de droite de l'équation de modes LP
        idx = np.argwhere(np.diff(np.sign(left - right))).flatten()
        # Calcule les valeurs de u correspondant à l'indice
        u_values = get_intersects(u, idx, V)
    return u_values, [left, right], f'l = {l}; {len(u_values)} intersects: {u_values}'


# Trouver les valeurs d'indice de réfraction effective et de gamma en fonction de u
def get_ref_index_gamma_values(l, intersects):
    ref_index = []
    core_pow = []
    st = ''
    for m in range(len(intersects)):
        # Calculer n_eff
        n_eff = np.sqrt((k0**2 * n1**2 - (intersects[m] / a)**2)/(k0**2))
        ref_index.append(n_eff)
        # Calculer puissance coeur
        gamma = get_pow(l, intersects[m])
        core_pow.append(gamma)
        st += f'LP_{l}{m+1} : n_eff = {n_eff}; gamma = {gamma}\n'
    return ref_index, core_pow, st

# Calculer la puissance contenue dans le coeur
def get_pow(l, u):
    w = np.sqrt(V**2 - u**2)
    gamma = 1 - (u**2 / V**2) * (1 - psi(l, w))
    return gamma


# Fonction psi
def psi(l, w):
    return (kv(l, w))**2 / (kv(l+1, w) * kv(l-1, w))


# Boucle principale
if __name__ == '__main__':
    l = 0
    while True:
        u_values = get_u(u, w, l, V)
        if not u_values[0]:
            break
        print(u_values[2])
        print(get_ref_index_gamma_values(l, u_values[0])[2])
        plt.figure(l)
        plt.plot(u, u_values[1][0])
        plt.plot(u, u_values[1][1])
        plt.title(f'l = {l}')
        plt.ylim(-2, 5)
        plt.xlim(0, round(V + 3))
        l += 1

    plt.show()
