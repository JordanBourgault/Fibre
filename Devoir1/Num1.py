from scipy.special import jv, kv
import numpy as np
import matplotlib.pyplot as plt

# Parameters
NA = 0.14
wl = 488e-9
a = 4.1e-6
k0 = 2 * np.pi / wl
V = k0 * NA * a
n1 = 1.4630

u = np.linspace(0, 15, 100000)
w = np.sqrt(V**2 - u**2)


def bessel_functions(l, u, w):
    with np.errstate(divide='ignore', invalid='ignore'):
        return jv(l, u) / (u * jv(l-1, u)), -kv(l, w) / (w * kv(l-1, w))


def print_intersects(l, u, idx, V):
    intersects = []
    intersect_list = u[idx][::2]
    for i in range(len(intersect_list)):
        if intersect_list[i] > V:
            break
        if intersect_list[i] > 0:
            intersects.append(intersect_list[i])
    print(f'l = {l}; {len(intersects)} intersects: {intersects}')
    return intersects


def get_values(l, intersects):
    ref_index = []
    core_pow = []
    st = ''
    for m in range(len(intersects)):
        n_eff = np.sqrt((k0**2 * n1**2 - (intersects[m] / a)**2)/(k0**2))
        gamma = get_pow(l, intersects[m])
        ref_index.append(n_eff)
        core_pow.append(gamma)
        st += f'LP_{l}{m+1} : n_eff = {n_eff}; gamma = {gamma}\n'
    print(st)
    return ref_index


def get_pow(l, u):
    w = np.sqrt(V**2 - u**2)
    psi = (kv(l, w))**2 / (kv(l+1, w) * kv(l-1, w))
    gamma = 1 - (u**2 / V**2) * (1 - psi)
    return gamma


def get_u(u, w, l, V):
    left, right = bessel_functions(l, u, w)
    with np.errstate(divide='ignore', invalid='ignore'):
        idx = np.argwhere(np.diff(np.sign(left - right))).flatten()
    return print_intersects(l, u, idx, V)


if __name__ == '__main__':
    l = 0
    while True:
        u_values = get_u(u, w, l, V)
        if not u_values:
            break
        get_values(l, u_values)

        # plt.figure(l)
        # plt.plot(u, left)
        # plt.plot(u, right)
        # plt.title(f'l = {l}')
        # plt.ylim(-5, 5)

        l += 1