from Laboratoires.utils import read_txt_data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import linregress


def linear(x, m, b):
    return m * x + b


# 1.1
wl, A = read_txt_data('data/1.1.txt')
plt.figure()
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.plot(wl, A, color='black')
plt.xlim((min(wl), max(wl)))
ticks = np.linspace(min(wl), max(wl), 5)
plt.xticks(ticks)

plt.xlabel("Longueur d'onde [nm]")
plt.ylabel("Amplitude [dBm]")
plt.savefig('figs/1.1.pdf')

# 1.2
I, P = read_txt_data('data/1.2.txt')
regression_DFB = linregress(I[3:], P[3:])
seuil_laser = -regression_DFB.intercept / regression_DFB.slope
print('Figure 1.2')
print(f"Seuil de l'effet laser: {round(seuil_laser, 3)} mA")
print(f"La pente de puissance en fonction du courant est de {round(regression_DFB.slope, 3)} uW/mA")
print(f"La puissance maximale atteinte est de {max(P)} uW à un courant de {I[np.argmax(P)]}")
print()
I_theo = np.linspace(min(I), max(I), 500)
P_theo = linear(I_theo, regression_DFB.slope, regression_DFB.intercept)

plt.figure()
plt.plot(I_theo, P_theo, label='Fit linéaire')
plt.plot(I, P, '.', color='black', label='Données expérimentales')
plt.xlim((min(I), max(I)))
plt.ylim([-20, max(P)])
ticks = np.linspace(min(I), max(I), 6)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks)
plt.legend()
plt.xlabel("Courant [mA]")
plt.ylabel("Puissance [µW]")
plt.savefig('figs/1.2.pdf')

# 2.1
I, P = read_txt_data('data/2.1.txt')
regression_PUMP = linregress(I[7:], P[7:])
seuil_laser = -regression_PUMP.intercept / regression_PUMP.slope
print('Figure 2.1')
print(f"Seuil de l'effet laser: {round(seuil_laser, 3)} mA")
print(f"La pente de puissance en fonction du courant est de {round(regression_PUMP.slope, 3)} uW/mA")
print(f"La puissance maximale atteinte est de {max(P)} uW à un courant de {I[np.argmax(P)]}")
print()
I_theo = np.linspace(min(I), max(I), 500)
P_theo = linear(I_theo, regression_PUMP.slope, regression_PUMP.intercept)

plt.figure()
plt.plot(I_theo, P_theo, label='Fit linéaire')
plt.plot(I, P, '.', color='black', label='Données expérimentales')
plt.xlim((min(I), max(I)))
plt.ylim([-4, max(P)])
ticks = np.linspace(min(I), max(I), 7)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks)
plt.legend()
plt.xlabel("Courant [mA]")
plt.ylabel("Puissance [mW]")
plt.savefig('figs/2.1.pdf')

# 2.2
wl, A = read_txt_data('data/2.2.txt')
plt.figure()
plt.plot(wl, A, color='black')
plt.xlim((min(wl), max(wl)))
ticks = np.linspace(min(wl), max(wl), 5)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks, np.round(ticks, 1))

plt.xlabel("Longueur d'onde [nm]")
plt.ylabel("Amplitude [dBm]")
plt.savefig('figs/2.2.pdf')

# 3
V, P = read_txt_data('data/3.txt')
plt.figure()
plt.plot(V, P, '.', color='black')
plt.xlim((min(V), max(V)))
ticks = np.linspace(min(V), max(V), 5)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks, np.round(ticks, 1))

plt.xlabel("Tension [V]")
plt.ylabel("Puissance sortie EVOA [µW]")
plt.savefig('figs/3.pdf')

# 3.1
attenuation = 10 * np.log10(P / 790)
plt.figure()
plt.plot(V, attenuation, '.', color='black')
plt.xlim((min(V), max(V)))
ticks = np.linspace(min(V), max(V), 5)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks, np.round(ticks, 1))

plt.xlabel("Tension [V]")
plt.ylabel("Atténuation de l'EVOA [dB]")
plt.savefig('figs/3.1.pdf')

# 5.1
wl, P_50, P_75, P_100, P_179 = read_txt_data('data/5.1.txt')
plt.figure()
plt.plot(wl, P_50, label='50 mA')
plt.plot(wl, P_75, label='75 mA')
plt.plot(wl, P_100, label='100 mA')
plt.plot(wl, P_179, label='178.8 mA')
plt.xlim((1500, 1580))
plt.xlabel("Longueur d'onde [nm]")
plt.ylabel("Puissance optique [mW/nm]")
plt.legend()
plt.savefig('figs/5.1.pdf')

# 5 absorption
P_out = 5.1
P_in = 67.8
beta = 0.8511
l = 20
absorption = - np.log(P_out / (beta * P_in)) / l
print('5')
print(f"Le coefficient d'absorption est {round(absorption, 3)} m^-1")

# 6.1
wl, A = read_txt_data('data/6.1.txt')
plt.figure()
plt.plot(wl, A, color='black')
plt.xlim((min(wl), max(wl)))
ticks = np.linspace(min(wl), max(wl), 5)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks, np.round(ticks, 1))

plt.xlabel("Longueur d'onde [nm]")
plt.ylabel("Amplitude [dBm]")
plt.savefig('figs/6.1.pdf')

# 6.2
I, P = read_txt_data('data/6.2.txt')
P_0 = -35
P -= P_0
plt.figure()
plt.plot(I, P, '.', color='black')
plt.xlim((min(I) - 5, max(I) + 5))
ticks = np.linspace(min(I), max(I), 6)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks)

plt.xlabel("Courant [mA]")
plt.ylabel("Gain à faibre signal [dB]")
plt.savefig('figs/6.2.pdf')

# 7.1
wl, A = read_txt_data('data/7.1.txt')
plt.figure()
plt.plot(wl, A, color='black')
plt.xlim((min(wl), max(wl)))
ticks = np.linspace(min(wl), max(wl), 5)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks, np.round(ticks, 1))

plt.xlabel("Longueur d'onde [nm]")
plt.ylabel("Amplitude [dBm]")
plt.savefig('figs/7.1.pdf')

# 7.2
A, V_0_178, V_35_178, V_43_178, V_5_178 = read_txt_data('data/7.2_178.txt')
A, V_0_55, V_35_55, V_43_55, V_5_55 = read_txt_data('data/7.2_55.txt')
tension = [0, 3.5, 4.3, 5]

P_pump_178 = linear(178, regression_PUMP.slope, regression_PUMP.intercept)
P_pump_55 = linear(55, regression_PUMP.slope, regression_PUMP.intercept)
P_in_178 = []
P_in_55 = []
for DFB_current in A:
    P_DFB = linear(DFB_current, regression_DFB.slope, regression_DFB.intercept)
    P_in_178.append(10 * np.log10((P_DFB + P_pump_178) * 1e-6 / 1e-3))
    P_in_55.append(10 * np.log10((P_DFB + P_pump_55) * 1e-6 / 1e-3))

plt.figure()
plt.plot(P_in_178, V_0_178, 'o-', color='C0', label=r'$V_{EVOA} = 0.0 V$')
plt.plot(P_in_178, V_35_178, 'o-', color='C1', label=r'$V_{EVOA} = 3.5 V$')
plt.plot(P_in_178, V_43_178, 'o-', color='C2', label=r'$V_{EVOA} = 4.3 V$')
plt.plot(P_in_178, V_5_178, 'o-', color='C3', label=r'$V_{EVOA} = 5.0 V$')

plt.plot(P_in_55, V_0_55, 'o:', color='C0', label=r'$V_{EVOA} = 0.0 V$')
plt.plot(P_in_55, V_35_55, 'o:', color='C1', label=r'$V_{EVOA} = 3.5 V$')
plt.plot(P_in_55, V_43_55, 'o:', color='C2', label=r'$V_{EVOA} = 4.3 V$')
plt.plot(P_in_55, V_5_55, 'o:', color='C3', label=r'$V_{EVOA} = 5.0 V$')

custom_lines = [Line2D([0], [0], color='k', linestyle='-'),
                Line2D([0], [0], color='k', linestyle=':'),
                Line2D([0], [0], color='C0', marker='o', lw=0),
                Line2D([0], [0], color='C1', marker='o', lw=0),
                Line2D([0], [0], color='C2', marker='o', lw=0),
                Line2D([0], [0], color='C3', marker='o', lw=0)]
plt.legend(custom_lines,
           ['Pompe 178 mA', 'Pompe 55 mA',
            r'$V_{EVOA} = 0.0 V$', r'$V_{EVOA} = 3.5 V$',
            r'$V_{EVOA} = 4.3 V$', r'$V_{EVOA} = 5.0 V$'],
           ncol=3,
           loc='lower center')
plt.ylim([-35, 10])
plt.xlabel('Puissance en entrée [dBm]')
plt.ylabel('Puissance en sortie [dBm]')
plt.savefig('figs/7.2.pdf')

# 7.2 gain

plt.figure()
plt.plot(P_in_178, V_0_178 - P_in_178, 'o-', color='C0', label=r'$V_{EVOA} = 0.0 V$')
plt.plot(P_in_178, V_35_178 - P_in_178, 'o-', color='C1', label=r'$V_{EVOA} = 3.5 V$')
plt.plot(P_in_178, V_43_178 - P_in_178, 'o-', color='C2', label=r'$V_{EVOA} = 4.3 V$')
plt.plot(P_in_178, V_5_178 - P_in_178, 'o-', color='C3', label=r'$V_{EVOA} = 5.0 V$')

plt.plot(P_in_55, V_0_55 - P_in_55, 'o:', color='C0', label=r'$V_{EVOA} = 0.0 V$')
plt.plot(P_in_55, V_35_55 - P_in_55, 'o:', color='C1', label=r'$V_{EVOA} = 3.5 V$')
plt.plot(P_in_55, V_43_55 - P_in_55, 'o:', color='C2', label=r'$V_{EVOA} = 4.3 V$')
plt.plot(P_in_55, V_5_55 - P_in_55, 'o:', color='C3', label=r'$V_{EVOA} = 5.0 V$')

custom_lines = [Line2D([0], [0], color='k', linestyle='-'),
                Line2D([0], [0], color='k', linestyle=':'),
                Line2D([0], [0], color='C0', marker='o', lw=0),
                Line2D([0], [0], color='C1', marker='o', lw=0),
                Line2D([0], [0], color='C2', marker='o', lw=0),
                Line2D([0], [0], color='C3', marker='o', lw=0)]
plt.legend(custom_lines,
           ['Pompe 178 mA', 'Pompe 55 mA',
            r'$V_{EVOA} = 0.0 V$', r'$V_{EVOA} = 3.5 V$',
            r'$V_{EVOA} = 4.3 V$', r'$V_{EVOA} = 5.0 V$'],
           ncol=3,
           loc='lower center')
plt.ylim([-30, 20])
plt.xlabel('Puissance en entrée [dBm]')
plt.ylabel('Gain [dB]')
plt.savefig('figs/7.2_gain.pdf')

print()

plt.show()

