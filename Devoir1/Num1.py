from scipy.special import jv, kv, jn_zeros
from scipy.integrate import solve_ivp, odeint
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
with np.errstate(divide='ignore', invalid='ignore'):
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
    return np.array(intersects)


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
def get_ref_index(l, u):
    # Calculer n_eff
    return np.sqrt((k0**2 * n1**2 - (u / a)**2)/(k0**2))

# Calculer la puissance contenue dans le coeur
def get_gamma(l, u):
    w = np.sqrt(V**2 - u**2)
    return 1 - (u**2 / V**2) * (1 - psi(l, w))


# Fonction psi
def psi(l, w):
    return (kv(l, w))**2 / (kv(l+1, w) * kv(l-1, w))


def differential_model(V, u):
    w = np.sqrt(V ** 2 - u ** 2)
    val = (u / V) * (1 - (w / (w+1)))
    return val

# Differential equation approach
def differential(V):
    u = solve_ivp(differential_model, [0, V], [0])
    return u


# Formule approchée de Miyagi
def miyagi(V):
    u_arr = []
    l = 0
    while True:
        m = 1
        sub_u = []
        while True:
            u_inf = jn_zeros(l, m)[-1]
            u = u_inf * (V / (V+1)) * (1 - u_inf**2 / (6 * (V+1)**3) - u_inf**4 / (20 * (V+1)**5))
            if u > V:
                u_arr.append(sub_u)
                break
            sub_u.append(u)
            m += 1
        if m == 1:
            break
        l += 1
    return u_arr


print(differential(V))


# Boucle principale
if __name__ == '__main__':
    l = 0
    print('----- Équation caractéristique -----')
    while True:
        u_values = get_u(u, w, l, V)
        if not u_values[0].any():
            break

        for m, u_val in enumerate(u_values[0], start=1):
            print(f'LP_{l}{m}: u = {u_val}; n_eff = {get_ref_index(l, u_val)}; gamma = {get_gamma(l, u_val)}')
        print('\n', end='')

        plt.figure(l)
        plt.plot(u, u_values[1][0])
        plt.plot(u, u_values[1][1])
        plt.title(f'l = {l}')
        plt.ylim(-2, 5)
        plt.xlim(0, round(V + 3))
        l += 1

    print('----- formule approchée de Miyagi -----')
    u_miyagi = miyagi(V)
    for l, u_arr in enumerate(u_miyagi, start=0):
        for m, u_val in enumerate(u_arr, start=1):
            print(f'LP_{l}{m}: u = {u_val}; n_eff = {get_ref_index(l, u_val)}')
        print('\n', end='')

    plt.show()
