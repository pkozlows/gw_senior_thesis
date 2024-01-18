import numpy as np
import matplotlib.pyplot as plt
import os
# import of my numpy files in the same directory
os.chdir('/Users/patrykkozlowski/caltech/classes/caltech/f23/ch121b/gan/dft/dzvp/333/ka_point_switching/form_at_solute_points')
print(os.getcwd())
band_kpts = np.load('band_kpts.npy')
absolute_points = np.load('absolute_points.npy')
kpath = np.load('kpath.npy')
sp_points = np.load('sp_points.npy')
e_kn_2 = np.load('e_kn_2.npy')
nbands = np.load('nbands.npy')

# now go forward with the potting
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
plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin//3, ymax=emax)
plt.xlabel('k-vector')
plt.ylabel('Fermi energy [eV]')

        
# Get the energy of band 18 at the Γ point
k_idx = np.argmin(np.linalg.norm(band_kpts - np.array([sp_points[3], 0, 0]), axis=1))
energy_band18_at_Gamma = e_kn_2[k_idx, 18] * au2ev
print(band_kpts[k_idx])
print(e_kn_2[k_idx, 18])
# change to the absolute points
# Get the energy of band 18 at the Γ point
absolute_index = np.argmin(np.linalg.norm(absolute_points - np.array([sp_points[3], 0, 0]), axis=1))
absolute_energy_eighteenth_at_gamma = e_kn_2[absolute_index, 18] * au2ev
print(absolute_points[absolute_index])
print(e_kn_2[absolute_index, 18])

# Plot a vertical line from Fermi level to band 18 at Γ point
plt.plot([sp_points[3], sp_points[3]], [0, energy_band18_at_Gamma], color='yellow', linestyle='--')
# plt.plot([sp_points[3], sp_points[3]], [0, absolute_energy_eighteenth_at_gamma], color='green', linestyle='--')
# Calculate and annotate the energy difference
energy_difference = abs(energy_band18_at_Gamma)  # Absolute value to ensure positive difference
mid_point_energy = energy_band18_at_Gamma / 2  # Midpoint between Fermi energy and band 18 energy

# plt.annotate(f'Energy: {energy_difference:.2f} eV',
#              xy=(sp_points[3], mid_point_energy),
#              xytext=(sp_points[3] + 0.3, mid_point_energy),
#              arrowprops=dict(facecolor='black', arrowstyle='->'),
#              horizontalalignment='left',
#              verticalalignment='center')
# print(f"Energy of the red band at Γ point: {energy_band18_at_Gamma:.2f} eV")

plt.savefig('dzvp_333_kpoints.png')


# # Find the energy of the conduction band (red band) at the special point
# energy_conduction = e_kn_2[k_idx, 18] * au2ev

# # Since the Fermi energy is 0, you just need the conduction band energy
# band_gap = energy_conduction

# # Add annotation for the bandgap
# gap_center = band_gap / 2  # This is for positioning the text at the center of the band gap
# plt.annotate(f'Band Gap: {band_gap:.2f} eV', 
            #  xy=(sp_points[3], gap_center), 
            #  xytext=(sp_points[3] + 0.2, gap_center),  # Position of the text
            #  arrowprops=dict(facecolor='black', arrowstyle='->'), 
            #  horizontalalignment='left', 
            #  verticalalignment='center')


    
# Print the energies
# print(f"Energy of band {band_idx_1} at k-point {sp_points[3]}: {energy_1:.2f} eV")
# print(f"Energy of band {band_idx_2} at k-point {sp_points[3]}: {energy_2:.2f} eV")
# Plot vertical green line showing the energy difference at gamma point
# plt.plot([sp_points[3], sp_points[3]], [gamma_energy_1, gamma_energy_2], color='green', linestyle='-')



