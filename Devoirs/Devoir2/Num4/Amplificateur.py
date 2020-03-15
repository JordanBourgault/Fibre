#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Amplificateur
authors: Pascal Paradis & Frédéric Maes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize as op

from Devoirs.Devoir1.Num1 import miyagi, get_gamma
from Devoirs.Devoir1.Num4 import sellmeier, SiO2, GeO2


# Sections efficaces
cs_s_abs_data = np.genfromtxt("Num4/abs_ErALP_1550nm.csv")
cs_s_ems_data = np.genfromtxt("Num4/ems_ErALP_1550nm.csv")

# Interpolation des sections efficaces
cs_s_abs = interp1d(cs_s_abs_data[:, 0], cs_s_abs_data[:, 1], fill_value="extrapolate")
cs_s_ems = interp1d(cs_s_ems_data[:, 0], cs_s_ems_data[:, 1], fill_value="extrapolate")


class Amplificateur(object):
    """Ceci est la classe amplificateur. Elle contient l'ensemble des
       paramètres de l'amplificateur ainsi que de l'ion Er3+
    """
    # définition des constantes utiles
    h = 6.626e-34   # Planck
    c = 3e8         # Vitesse de la lumière

    def __init__(self, L, lambda_s, pump="clad"):
        """Constructeur de la classe Amplificateur
        """

        #1) On commence avec les paramètres qui sont constant dans
        #   l'amplificateur simulé
        self.Ntot = 1e26                    # Densité ioniques de dopant [m^3]
        self.ac = 8e-6 / 2                  # Rayon du coeur
        self.ag = 125e-6 / 2                # Rayon de la gaine
        self.x_germ = 0.045                 # Concentration de germanium coeur
        self.lambda_p = 978e-9              # Longeur d'onde de pompage
        self.w22 = 1.7e-23                  # Transfert par pair d'ion
        self.tau2 = 10.8e-3                 # Temps de vie du niveau 2
        self.tau3 = 0.005e-3                # Temps de vie du niveau 3
        self.tau4 = 0.005e-3                # Temps de vie du niveau 4
        self.sigma_p = 31.2e-26             # Section efficace d'absorption à la longueur d'onde de la pompe
        self.sigma_esa = 6.2e-26            # Section efficace du ESA
        self.alpha_p = 0.01                 # Pertes de la fibre à la longueur d'onde de pompage
        self.alpha_s = 0.01                 # Pertes de la fibre à la longueur d'onde du signal

        # 2) Pour les paramètres ci-dessous , nous leur attribuons la valeur donnée au constructeur
        self.L = L                          # longueur de l'amplificateur
        self.lambda_s = lambda_s            # Longueur d'onde du signal.

        # 3) Les paramètres qui dépendent des paramètres définis précédemment à l'étape 1) et 2).
        self.Ac = np.pi * self.ac**2        # Aire coeur
        self.Ag = np.pi * self.ag**2        # Aire de la gaine
        # Calcul du confinement de la pompe selon qu'elle est injectée dans la gaine ou dans le coeur de la fibre
        if pump == "clad":
            self.gamma_p = self.Ac / self.Ag            # Confinement de la pompe dans la gaine
        elif pump == "core":
            self.gamma_p = self.confinement(self.lambda_p, self.x_germ)     # Confinement de la pompe dans le coeur
        # Confinement du signal
        self.gamma_s = self.confinement(self.lambda_s, self.x_germ)         # Confinement du signal dans le coeur.
        # Section efficace d'absorption du signal
        self.sigma_abs = cs_s_abs(self.lambda_s)
        # Section efficace d'émission du signal
        self.sigma_ems = cs_s_ems(self.lambda_s)
        # Autres propriétés nécessaire pour le bon fonctionnement du code.
        self.N = np.ones(shape=(1, 4)) * 0.25 * self.Ntot
        self.Pp = 0
        self.Ps = 0
        self.rp = 0
        self.rs = 0
        self.resa = 0
        self.dz = 0
        self.num_elements = 501

    def confinement(self,  lambda_i, x_germ):
        n1 = sellmeier(SiO2, GeO2, x_germ, lambda_i)
        n2 = sellmeier(SiO2, GeO2, 0, lambda_i)
        NA = np.sqrt(n1**2 - n2**2)
        k0 = 2 * np.pi / lambda_i
        V = k0 * NA * self.ac
        u = miyagi(V, True)
        return get_gamma(0, u, V)

    def sol(self, Pp_launch, Ps_launch, num_elements):
        """Cette fonction intègre les équations de puissance élément par élément dans
        l'amplificateur pour une puissance injectée de pompe et de signal donnée.

        Inputs:
               Pp_launch(float):   Puissance pompe injectée (W)
               Ps_launch(float):   Puissance du signal injectée (W)
               num_elements(int):  Discrétization de la fibre dopée

        Outputs:
               z_sol(vect ix1):    Vecteur discrétisant la longueur de l'ampli
               Pp_sol(vect ix1):   Solution de la puissance pompe à
                                   chaque position de z_sol
               Ps_sol(vect ix1):   Solution de la puissance du signal à
                                   chaque position de z_sol
               N_sol(vect ix4):    Populations des niveaux d'énergie à chaque
                                   position de z_sol.
        """

        # Initialisation
        self.num_elements = num_elements
        z_sol = np.linspace(0, self.L, num_elements)
        self.dz = z_sol[1] - z_sol[0]
        Pp_sol = np.zeros(len(z_sol))
        Ps_sol = np.zeros(len(z_sol))
        N_sol = np.zeros((len(z_sol), 4))

        # assignation des valeurs à z=0
        Pp_sol[0] = Pp_launch
        Ps_sol[0] = Ps_launch
        N_sol[0,:] = self.sol_eq_niv(Pp_launch, Ps_launch)
        # Itérations de 0 à L pour trouver les solutions des équations de niveau
        # et de puissance
        for i in range(1, len(z_sol)):
            self.Pp = self._Pp() # Ici on intègre la puissance précédente pour obtenir la puissance suivante
            self.Ps = self._Ps()
            Pp_sol[i] = self.Pp
            Ps_sol[i] = self.Ps
            N_sol[i,:] = self.sol_eq_niv(self.Pp, self.Ps)
        return z_sol, Pp_sol, Ps_sol, N_sol

    def eq_niv(self, N):
        """Cette fonction contient les équations des niveaux (rate equations)
        de l'erbium en régime permanent.

        Inputs:
            N(vect 1x4):     Vecteur solution qui entraine un résidu nul (F=0).
            Pp (float):      Puissance de la pompe [W]
            Ps (float):      Puissance du signal laser [W]

        Outputs:
            F (vect 1x4):   Résidu des équations de niveaux en régime permanent
         """
        self.N = N
        self._Rs()
        self._Rp()
        self._Resa()
        return (self._dN4(),
                self._dN3(),
                self._dN2(),
                self._zero())

    def sol_eq_niv(self, Pp, Ps):
        """Cette fonction résoud les équations de niveaux avec une puissance de
        pompe et de signal donnée.

        Inputs
            Pp (float):  Puissances de la pompe[W]
            Ps (float):  Puissances du signal laser [W]

        Outputs
            N (vect 1x4):   Densité atomiques des niveaux d'énergie [m^-3]

        Les équations des niveaux sont un ensemble d'équations non-linéaires à
        4 variables.
        """
        self.Pp = Pp
        self.Ps = Ps
        self.N = op.root(self.eq_niv, self.N).x
        return self.N

    # Les fonctions suivantes (méthodes de l'objet Amplificateur) régissent la physique de
    # l'amplificateur

    # Taux de pompage et laser
    def _Rp(self):
        self.rp = self.sigma_p * self.gamma_p * self.lambda_p / \
                  (self.h * self.c * self.Ac) * (self.N[0] - self.N[2]) * self.Pp
        return

    def _Resa(self):
        self.resa = self.sigma_esa * self.gamma_p * self.lambda_p * self.N[2] * self.Pp / \
                    (self.h * self.c * self.Ac)
        return

    def _Rs(self):
        self.rs = self.gamma_s * self.lambda_s * (self.sigma_ems * self.N[1] - self.sigma_abs * self.N[0]) * self.Ps / \
                  (self.h * self.c * self.Ac)
        return

    # Équations des niveaux

    def _dN4(self):
        return -self.N[3]/self.tau4 + self.resa

    def _dN3(self):
        return -self.N[2]/self.tau3 + self.N[3]/self.tau4 + self.rp - self.resa + self.w22 * self.N[1]**2

    def _dN2(self):
        return -self.N[1]/self.tau2 + self.N[2]/self.tau3 - self.rs - 2 * self.w22 * self.N[1]**2

    def _zero(self):
        return self.Ntot - sum(self.N)

    # Intégration de la puissance de pompe et du signal
    def _Pp(self):
        return self.Pp * np.exp(-(self.gamma_p*(self.sigma_p*self.N[0] + self.sigma_esa*self.N[2])
                                  + self.alpha_p) * self.dz)

    def _Ps(self):
        return self.Ps * np.exp((self.gamma_s * (self.sigma_ems * self.N[1] - self.sigma_abs * self.N[0])
                                 - self.alpha_p) * self.dz)

    def gamma(self):
        return self.sigma_ems * self.N[1] - self.sigma_abs * self.N[0]


if __name__ == '__main__':
    # Numéro 2
    plt.figure()
    for P_pump in [0, 2, 4, 8, 16, 32, 64, 128]:
        wavelength = np.linspace(1450e-9, 1600e-9, 500)
        gain = []
        for wl in wavelength:
            ampli = Amplificateur(1, wl, pump='clad')
            z_sol, Pp_sol, Ps_sol, N_sol = ampli.sol(P_pump, 0, 2)
            current_gain = ampli.gamma()
            gain.append(current_gain)

        print(P_pump)
        plt.plot(wavelength * 1e9, gain, label=f'{P_pump} W')

    plt.xlim([wavelength[0] * 1e9, wavelength[-1] * 1e9])
    plt.xlabel("Longueur d'onde [nm]")
    plt.ylabel("Gain [-]")
    plt.legend(title='P pompe:')
    plt.savefig('figs/Num4_2.pdf', bbox_inches='tight')

    # Numéro 3
    plt.figure()
    for P_pump in [1, 2, 4, 6, 8, 10]:
        gain = []
        P_sig = np.logspace(-9, 1, 150)
        for P in P_sig:
            wl = 1580e-9
            ampli = Amplificateur(5, wl, pump='clad')
            z_sol, Pp_sol, Ps_sol, N_sol = ampli.sol(P_pump, P, 101)
            gain.append(0.5 * np.log(Ps_sol[-1] / Ps_sol[0]))

        print(P_pump)
        plt.semilogx(P_sig, gain, label=f'{P_pump} W')

    plt.xlim([P_sig[0], P_sig[-1]])
    plt.xlabel("Puissance du signal d'entrée [W]")
    plt.ylabel('Gain G [dB]')
    plt.legend(title='P pompe:')
    plt.savefig('figs/Num4_3.pdf', bbox_inches='tight')
    plt.show()

    # Numéro 4
    length = np.linspace(0, 15, 200)
    gain_core = []
    gamma_core = []
    abs_core = []

    gain_clad = []
    gamma_clad = []
    abs_clad = []

    for L in length:
        ampli_core = Amplificateur(L, 1540e-9, pump='core')
        z_sol_core, Pp_sol_core, Ps_sol_core, N_sol_core = ampli_core.sol(0.75, 1e-6, 301)
        gain_core.append(0.5 * np.log(Ps_sol_core[-1] / Ps_sol_core[0]))
        gamma_core.append(ampli_core.gamma())
        abs_core.append(0.5 * np.log(Pp_sol_core[-1] / Pp_sol_core[0]))

        ampli_clad = Amplificateur(L, 1540e-9, pump='clad')
        z_sol_clad, Pp_sol_clad, Ps_sol_clad, N_sol_clad = ampli_clad.sol(10, 1e-6, 301)
        gain_clad.append(0.5 * np.log(Ps_sol_clad[-1] / Ps_sol_clad[0]))
        gamma_clad.append(ampli_clad.gamma())
        abs_clad.append(0.5 * np.log(Pp_sol_clad[-1] / Pp_sol_clad[0]))
        print(L)

    plt.figure()
    plt.plot(length, gain_core, label='Coeur')
    plt.plot(length, gain_clad, label='Gaine')
    plt.legend(title='Type de pompage:')
    plt.xlabel('Longueur de la cavité [m]')
    plt.ylabel('Gain G [dB]')
    plt.xlim([length[0], length[-1]])
    plt.savefig('figs/Num4_4_G.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(length, gamma_core, label='Coeur')
    plt.plot(length, gamma_clad, label='Gaine')
    plt.legend(title='Type de pompage:')
    plt.xlabel('Longueur de la cavité [m]')
    plt.ylabel(r'Coefficient de gain $\gamma$ [-]')
    plt.xlim([length[0], length[-1]])
    plt.savefig('figs/Num4_4_gamma.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(length, abs_core, label='Coeur')
    plt.plot(length, abs_clad, label='Gaine')
    plt.legend(title='Type de pompage:')
    plt.xlabel('Longueur de la cavité [m]')
    plt.ylabel(r'Absorption de la pompe [dB]')
    plt.xlim([length[0], length[-1]])
    plt.savefig('figs/Num4_4_abs.pdf', bbox_inches='tight')

    plt.show()
