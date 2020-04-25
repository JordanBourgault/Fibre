from Laboratoires.utils import read_txt_data
import matplotlib.pyplot as plt
import numpy as np


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
plt.figure()
plt.plot(I, P, color='black')
plt.xlim((min(I), max(I)))
ticks = np.linspace(min(I), max(I), 6)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks)

plt.xlabel("Courant [mA]")
plt.ylabel("Puissance [µW]")
plt.savefig('figs/1.2.pdf')

# 2.1
I, P = read_txt_data('data/2.1.txt')
plt.figure()
plt.plot(I, P, color='black')
plt.xlim((min(I), max(I)))
ticks = np.linspace(min(I), max(I), 7)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks)

plt.xlabel("Courant [mA]")
plt.ylabel("Puissance [µW]")
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
plt.plot(V, P, color='black')
plt.xlim((min(V), max(V)))
ticks = np.linspace(min(V), max(V), 5)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks, np.round(ticks, 1))

plt.xlabel("Tension [V]")
plt.ylabel("Puissance sortie EVOA [µW]")
plt.savefig('figs/3.pdf')

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
plt.figure()
plt.plot(I, P, color='black')
plt.xlim((min(I), max(I)))
ticks = np.linspace(min(I), max(I), 6)
plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
plt.xticks(ticks)

plt.xlabel("Courant [mA]")
plt.ylabel("Puissance [dBm]")
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




plt.show()

