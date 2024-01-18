import numpy as np
import matplotlib.pyplot as plt
import os

# import of my numpy files in the same directory
os.chdir('/Users/patrykkozlowski/caltech/classes/caltech/f23/ch121b/gan/dft/dzvp/333')
print(os.getcwd())
band_kpts = np.load('band_kpts.npy')
kpath = np.load('kpath.npy')
sp_points = np.load('sp_points.npy')
e_kn_2 = np.load('e_kn_2.npy')
nbands = np.load('nbands.npy')

# plod all of the bands with blue lines
au2ev = 27.21139
emin = -1*au2ev
emax = 1*au2ev
plt.figure(figsize=(5, 6))
for n in range(nbands):
    plt.plot(kpath, [e[n]*au2ev for e in e_kn_2], color='#87CEEB')
    
# Plot band 18 with a red line
plt.plot(kpath, e_kn_2[:, 18] * au2ev, color='red')
for p in sp_points:
    plt.plot([p, p], [emin, emax], 'k-')
plt.plot([0, sp_points[-1]], [0, 0], 'k-')
plt.xticks(sp_points, ['$%s$' % n for n in [r'\Gamma', 'M', 'K', r'\Gamma', 'A', 'L', 'H', 'A']])
plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin//5, ymax=emax//2)
plt.xlabel('k-vector')
plt.ylabel('Fermi energy [eV]')

        
# Get the energy of band 18 at the Γ point
print(sp_points[3])
gamma_point_indices =[index for index, point in enumerate(band_kpts) if all(element == 0 for element in point)]
assert(len(gamma_point_indices) == 2)


k_idx = gamma_point_indices[1]
energy_band18_at_Gamma = e_kn_2[k_idx, 18] * au2ev
print(band_kpts[k_idx])
print(e_kn_2[k_idx, 18])
# Plot a vertical line from Fermi level to band 18 at Γ point
plt.plot([sp_points[3], sp_points[3]], [0, energy_band18_at_Gamma], color='yellow', linestyle='--')

# Calculate and annotate the energy difference
energy_difference = abs(energy_band18_at_Gamma)  # Absolute value to ensure positive difference
mid_point_energy = energy_band18_at_Gamma / 2  # Midpoint between Fermi energy and band 18 energy
plt.annotate(f'Energy: {energy_difference:.2f} eV',
             xy=(sp_points[3], mid_point_energy),
             xytext=(sp_points[3] + 0.3, mid_point_energy),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='left',
             verticalalignment='center')
print(f"Energy of the red band at Γ point: {energy_band18_at_Gamma:.2f} eV")

plt.savefig('dzvp_333_10-19.png')




